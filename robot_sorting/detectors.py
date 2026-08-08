import time
from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np

from robot_sorting import config as cfg
from robot_sorting.types import Detection
from robot_sorting.vision_gate import (
    bbox_centre,
    passes_opencv_gate,
)


class BaseDetector(ABC):
    """Interface to base all detectors on."""
    name = "base_detector"

    @abstractmethod
    def detect(self, image) -> Tuple[List[Detection], float]:
        """
        All detectors must implement this method.
        Returns:
            detections: list of valid Detection objects
            inference_time_ms: detector runtime in milliseconds
        """
        raise NotImplementedError


class OpenCVColourDetector(BaseDetector):
    """OpenCV Colour Detector."""
    name = "opencv_colour"

    def detect(self, image) -> Tuple[List[Detection], float]:
        # quick check to make sure the image is not empty
        if image is None:
            raise ValueError("Detector received an empty image.")

        # start timer to count inference time
        start = time.perf_counter()

        # convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # initialise detection list
        detections: List[Detection] = []

        # iterate through each colour range
        for colour in cfg.HSV_RANGES:
            # create mask
            mask = self._create_mask(hsv, colour)

            # apply region of interest
            mask = self._apply_roi(mask)

            # find the white blobs in the binary mask
            contours, _ = cv2.findContours(
                mask,
                # retrieve the external contours only
                cv2.RETR_EXTERNAL,
                # remove redundant points
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:
                # get the area of the contour
                area = float(cv2.contourArea(contour))

                # get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # check if the bounding box is valid
                valid, _reason = passes_opencv_gate(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area=area,
                )

                # if the bounding box is not valid, skip it
                if not valid:
                    continue

                # Center is calculated from the original unpadded contour box.
                # Padding is visual only and must not shift the grasp point.
                u, v = bbox_centre(x, y, w, h)

                # Add the detection to the list
                detections.append(
                    Detection(
                        method=self.name,
                        class_name=colour,
                        confidence=1.0,
                        bbox_x=x,
                        bbox_y=y,
                        bbox_w=w,
                        bbox_h=h,
                        u=float(u),
                        v=float(v),
                        area=area,
                    )
                )

        # sort detections by area in descending order
        detections.sort(
            key=lambda detection: detection.area,
            reverse=True,
        )

        # calculate inference time in milliseconds
        inference_time_ms = (
                                    time.perf_counter() - start
                            ) * 1000.0

        return detections, inference_time_ms

    # Static helper methods
    @staticmethod
    def _create_mask(hsv, colour):
        """Create and clean a binary mask for the selected colour from an HSV image."""

        # start with an empty mask full of zeros, therefore, black
        # Create a mask with the same height and width as the image.
        final_mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8,
        )

        # Loop through all configured HSV ranges for the current colour
        for lower, upper in cfg.HSV_RANGES[colour]:
            # Convert the colour range to an array of integers
            lower_array = np.array(lower, dtype=np.uint8)
            upper_array = np.array(upper, dtype=np.uint8)

            # Find matching pixels in the HSV image
            # if yes 255 (white) else 0 (black)
            colour_mask = cv2.inRange(
                hsv,
                lower_array,
                upper_array,
            )

            # update the final mask with the colour mask
            final_mask = cv2.bitwise_or(
                final_mask,
                colour_mask,
            )

        # Create kernels used to clean the binary mask
        open_kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        close_kernel = np.ones(
            (7, 7),
            dtype=np.uint8,
        )

        # Opening removes small isolated regions of noise
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_OPEN,
            open_kernel,
        )

        # Closing fills small gaps or holes inside detected regions
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        return final_mask

    @staticmethod
    def _apply_roi(mask):
        """Apply a region of interest (ROI) to the mask."""

        # Create a mask with the same size as the image filled with zeros (black)
        roi_mask = np.zeros_like(mask)

        # Fill the ROI with white pixels
        roi_mask[
            cfg.ROI_Y_MIN:cfg.ROI_Y_MAX,
            cfg.ROI_X_MIN:cfg.ROI_X_MAX,
        ] = 255

        # Combine the colour mask and ROI
        return cv2.bitwise_and(
            mask,
            roi_mask,
        )
