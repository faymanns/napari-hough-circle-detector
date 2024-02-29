"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

import itertools
from typing import TYPE_CHECKING

import cv2
import numpy as np
from magicgui.widgets import (
    Container,
    FileEdit,
    Label,
    LineEdit,
    PushButton,
    Select,
    create_widget,
)

import napari_hough_circle_detector._utils as _utils

if TYPE_CHECKING:
    import napari


class CircleDetector(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._preprocessing_title = Label(label="Preprocessing (optional)")

        # use create_widget to generate widgets from type annotations
        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )

        self._median_filter_slider = create_widget(
            label="Median filter", annotation=int, widget_type="IntSlider"
        )
        self._median_filter_slider.min = 1
        self._median_filter_slider.max = 50

        self._circle_detection_title = Label(label="Circle detection")

        self._min_dist_slider = create_widget(
            label="Minimum distance", annotation=int, widget_type="IntSlider"
        )
        self._min_dist_slider.min = 1
        self._min_dist_slider.max = 200

        self._min_dist_slider = create_widget(
            label="Minimum distance", annotation=int, widget_type="IntSlider"
        )
        self._min_dist_slider.min = 1
        self._min_dist_slider.max = 200

        self._param1_slider = create_widget(
            label="Edge detection parameter",
            annotation=int,
            widget_type="IntSlider",
        )
        self._param1_slider.min = 10
        self._param1_slider.max = 1000

        self._param2_slider = create_widget(
            label="Accumulator threshold",
            annotation=int,
            widget_type="IntSlider",
        )
        self._param2_slider.min = 10
        self._param2_slider.max = 300

        self._min_radius_slider = create_widget(
            label="Minimum radius", annotation=int, widget_type="IntSlider"
        )
        self._min_radius_slider.min = 1
        self._min_radius_slider.max = 300

        self._max_radius_slider = create_widget(
            label="Maximum radius", annotation=int, widget_type="IntSlider"
        )
        self._max_radius_slider.min = 1
        self._max_radius_slider.max = 500

        self._export_title = Label(label="Export")

        self._file_type_selection = Select(
            name="File type",
            choices=("csv", "mask"),
            value="csv",
        )
        self._export_directory = FileEdit(label="Directory", mode="d")
        self._export_file = LineEdit(label="File", value="circles.csv")
        self._export_button = PushButton(label="Export")

        # connect your own callbacks
        self._image_layer_combo.changed.connect(self._detect_circles)
        self._median_filter_slider.changed.connect(self._median_filter)
        self._min_dist_slider.changed.connect(self._detect_circles)
        self._param1_slider.changed.connect(self._detect_circles)
        self._param2_slider.changed.connect(self._detect_circles)
        self._min_radius_slider.changed.connect(self._detect_circles)
        self._max_radius_slider.changed.connect(self._detect_circles)

        # append into/extend the container with your widgets
        self.extend(
            [
                self._preprocessing_title,
                self._image_layer_combo,
                self._median_filter_slider,
                self._circle_detection_title,
                self._min_dist_slider,
                self._param1_slider,
                self._param2_slider,
                self._min_radius_slider,
                self._max_radius_slider,
                self._export_title,
                self._file_type_selection,
                self._export_directory,
                self._export_file,
                self._export_button,
            ]
        )

    def _median_filter(self):
        """
        Median filters the input image with kernel size `2 * median_filter_strength - 1` and returns
        a napari LayerDataTuple that creates a new image layer named 'Median filtered' or replaces
        image of a layer with that name.
        """
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return
        name = image_layer.name + "_median_filtered"

        median_filter_size = self._median_filter_slider.value * 2 - 1

        img = _utils._extract_img_data(image_layer, view_only=False)
        filtered = np.zeros_like(img)
        ndim = img.ndim
        if ndim == 2:
            filtered = cv2.medianBlur(img, median_filter_size)
        else:
            for indices in itertools.product(
                *[range(n) for n in img.shape[:-2]]
            ):
                filtered[indices] = cv2.medianBlur(
                    img[indices], median_filter_size
                )

        if name in self._viewer.layers:
            self._viewer.layers[name].data = filtered
        else:
            self._viewer.add_image(filtered, name=name)

    def _detect_circles(self):
        """
        Computes and edge map using canny edge detection and detects circles using the
        Hough transform. For more information see the documentation of opencv's
        HoughCircles function.

        """
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        img = _utils._extract_img_data(image_layer, view_only=False)
        contrast_limits = image_layer.contrast_limits
        name = image_layer.name

        img = np.clip(img, *contrast_limits)
        img = (
            (img - contrast_limits[0])
            / (contrast_limits[1] - contrast_limits[0])
            * 255
        )
        img = img.astype(np.uint8)
        edges = cv2.Canny(
            img, self._param1_slider.value / 2, self._param1_slider.value
        )
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            self._min_dist_slider.value,
            param1=self._param1_slider.value,
            param2=self._param2_slider.value,
            minRadius=self._min_radius_slider.value,
            maxRadius=self._max_radius_slider.value,
        )
        circles = np.zeros((1, 3)) if circles is None else circles[0]

        name_edges = name + "_edges"
        if name_edges in self._viewer.layers:
            self._viewer.layers[name_edges].data = edges
        else:
            self._viewer.add_image(edges, name=name_edges)

        name_circles = name + "_circles"
        if name_circles in self._viewer.layers:
            self._viewer.layers[name_circles].data = circles[:, (1, 0)]
            self._viewer.layers[name_circles].size = circles[:, 2] * 2
        else:
            n = circles.shape[0]
            self._viewer.add_points(circles[:, (1, 0)], name=name_circles)
            self._viewer.layers[name_circles].size = circles[:, 2] * 2
            self._viewer.layers[name_circles].edge_color = ["red"] * n
            self._viewer.layers[name_circles].face_color = [[0] * 4] * n
