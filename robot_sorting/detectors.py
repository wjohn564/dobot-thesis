import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from robot_sorting import config as cfg
from robot_sorting.types import Detection
from robot_sorting.vision_gate import (
    bbox_centre,
    passes_opencv_gate,
    passes_workspace_gate,
)


class BaseDetector(ABC):
    """Shared detector interface."""

    name = "base_detector"

    @abstractmethod
    def detect(self, image) -> Tuple[List[Detection], float]:
        """Return detections and inference time in milliseconds."""
        raise NotImplementedError

    def warm_up(
            self,
            image,
            runs: int = 3,
    ):
        """Run the detector a few times before experiment timing."""

        for _ in range(runs):
            self.detect(image)


class OpenCVColourDetector(BaseDetector):
    """OpenCV Colour Detector."""

    name = "opencv_colour"

    def detect(self, image) -> Tuple[List[Detection], float]:

        # Reject empty images.
        if image is None:
            raise ValueError(
                "Detector received an empty image."
            )

        # Time the detector call.
        start = time.perf_counter()

        # Convert to HSV for colour segmentation.
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        # Store valid detections.
        detections: List[Detection] = []

        # Process each configured colour.
        for colour in cfg.HSV_RANGES:

            # Create and crop the colour mask.
            mask = self._create_mask(
                hsv,
                colour,
            )

            mask = self._apply_roi(mask)

            # Find separate blobs in the mask.
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:

                # Measure the contour and bounding box.
                area = float(
                    cv2.contourArea(contour)
                )

                x, y, w, h = cv2.boundingRect(
                    contour
                )

                # Apply the OpenCV detection gate.
                valid, _reason = passes_opencv_gate(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area=area,
                )

                if not valid:
                    continue

                # Use the unpadded box centre for grasping.
                u, v = bbox_centre(
                    x,
                    y,
                    w,
                    h,
                )

                # Store the valid detection.
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

        # Largest detections first.
        detections.sort(
            key=lambda detection: detection.area,
            reverse=True,
        )

        # Convert runtime to milliseconds.
        inference_time_ms = (
                                    time.perf_counter() - start
                            ) * 1000.0

        return detections, inference_time_ms

    @staticmethod
    def _create_mask(hsv, colour):
        """Create and clean a binary mask for one colour."""

        # Start with an empty mask.
        final_mask = np.zeros(
            hsv.shape[:2],
            dtype=np.uint8,
        )

        # Combine all HSV ranges for this colour.
        for lower, upper in cfg.HSV_RANGES[colour]:
            # Convert configured bounds to OpenCV arrays.
            lower_array = np.array(
                lower,
                dtype=np.uint8,
            )

            upper_array = np.array(
                upper,
                dtype=np.uint8,
            )

            # Keep pixels inside the HSV range.
            colour_mask = cv2.inRange(
                hsv,
                lower_array,
                upper_array,
            )

            # Merge this range into the final mask.
            final_mask = cv2.bitwise_or(
                final_mask,
                colour_mask,
            )

        # Morphological kernels for mask cleanup.
        open_kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        close_kernel = np.ones(
            (7, 7),
            dtype=np.uint8,
        )

        # Opening removes small noise.
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_OPEN,
            open_kernel,
        )

        # Closing fills small gaps.
        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        return final_mask

    @staticmethod
    def _apply_roi(mask):
        """Apply a region of interest (ROI) to the mask."""

        # Build the ROI mask.
        roi_mask = np.zeros_like(mask)

        roi_mask[
            cfg.ROI_Y_MIN:cfg.ROI_Y_MAX,
            cfg.ROI_X_MIN:cfg.ROI_X_MAX,
        ] = 255

        # Keep only pixels inside the ROI.
        return cv2.bitwise_and(
            mask,
            roi_mask,
        )


class YOLODetector(BaseDetector):
    """YOLO26n detector using the NCNN model."""

    name = "yolo26n"

    def __init__(
            self,
            model_path,
            confidence_threshold: float = 0.25,
    ):
        """Load the exported YOLO26n model."""

        # Import here so OpenCV does not depend on Ultralytics.
        from ultralytics import YOLO

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: "
                f"{self.model_path}"
            )

        self.confidence_threshold = float(
            confidence_threshold
        )

        print(
            f"Loading YOLO26n model: "
            f"{self.model_path}"
        )

        self.model = YOLO(
            str(self.model_path)
        )

    def detect(self, image) -> Tuple[List[Detection], float]:
        """Detect blocks using YOLO26n."""

        if image is None:
            raise ValueError(
                "Detector received an empty image."
            )

        # Time the complete detector call.
        start = time.perf_counter()

        results = self.model.predict(
            source=image,
            imgsz=640,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: List[Detection] = []

        if results:
            result = results[0]

            if result.boxes is not None:

                boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

                scores = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .numpy()
                )

                class_ids = (
                    result.boxes.cls
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                for box, score, class_id in zip(
                        boxes,
                        scores,
                        class_ids,
                ):
                    x1, y1, x2, y2 = [
                        float(value)
                        for value in box
                    ]

                    class_name = result.names[
                        int(class_id)
                    ]

                    # Ignore classes not used by the sorter.
                    if class_name not in cfg.COLOUR_TO_BIN:
                        continue

                    # Calculate the centre from the model box.
                    u = (
                                x1 + x2
                        ) / 2.0

                    v = (
                                y1 + y2
                        ) / 2.0

                    # Keep the same workspace check for every detector.
                    valid, _reason = (
                        passes_workspace_gate(
                            u,
                            v,
                        )
                    )

                    if not valid:
                        continue

                    bbox_x = int(
                        round(x1)
                    )

                    bbox_y = int(
                        round(y1)
                    )

                    bbox_w = max(
                        1,
                        int(
                            round(
                                x2 - x1
                            )
                        ),
                    )

                    bbox_h = max(
                        1,
                        int(
                            round(
                                y2 - y1
                            )
                        ),
                    )

                    # Learned detectors use bounding box area.
                    area = float(
                        (x2 - x1)
                        * (y2 - y1)
                    )

                    detections.append(
                        Detection(
                            method=self.name,
                            class_name=class_name,
                            confidence=float(score),
                            bbox_x=bbox_x,
                            bbox_y=bbox_y,
                            bbox_w=bbox_w,
                            bbox_h=bbox_h,
                            u=float(u),
                            v=float(v),
                            area=area,
                        )
                    )

        # Keep the same ordering used by OpenCV.
        detections.sort(
            key=lambda detection: detection.area,
            reverse=True,
        )

        inference_time_ms = (
                                    time.perf_counter() - start
                            ) * 1000.0

        return detections, inference_time_ms


class FasterRCNNDetector(BaseDetector):
    """Faster R-CNN MobileNetV3 320 detector."""

    name = "faster_rcnn_mobilenet_v3_320"

    # Faster R-CNN uses 0 for background.
    CLASS_NAMES = {
        1: "red",
        2: "blue",
        3: "yellow",
        4: "green",
    }

    def __init__(
            self,
            model_path,
            confidence_threshold: float = 0.20,
    ):
        """Load the trained Faster R-CNN model."""

        # Import here so other detectors do not need PyTorch.
        import torch

        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        self.torch = torch

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Faster R-CNN model not found: "
                f"{self.model_path}"
            )

        self.confidence_threshold = float(
            confidence_threshold
        )

        # Raspberry Pi inference runs on the CPU.
        self.device = torch.device(
            "cpu"
        )

        print(
            f"Loading Faster R-CNN model: "
            f"{self.model_path}"
        )

        # Build the same architecture used for training.
        self.model = (
            fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=5,
            )
        )

        state_dict = torch.load(
            str(self.model_path),
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def detect(self, image) -> Tuple[List[Detection], float]:
        """Detect blocks using Faster R-CNN."""

        if image is None:
            raise ValueError(
                "Detector received an empty image."
            )

        # Time the complete detector call.
        start = time.perf_counter()

        # OpenCV gives BGR images. Faster R-CNN uses RGB.
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image_tensor = (
                self.torch
                .from_numpy(rgb_image)
                .permute(2, 0, 1)
                .contiguous()
                .float()
                / 255.0
        )

        image_tensor = image_tensor.to(
            self.device
        )

        with self.torch.inference_mode():
            output = self.model(
                [image_tensor]
            )[0]

        boxes = (
            output["boxes"]
            .detach()
            .cpu()
            .numpy()
        )

        labels = (
            output["labels"]
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        scores = (
            output["scores"]
            .detach()
            .cpu()
            .numpy()
        )

        detections: List[Detection] = []

        for box, label, score in zip(
                boxes,
                labels,
                scores,
        ):
            # Reject low confidence detections.
            if score < self.confidence_threshold:
                continue

            class_name = self.CLASS_NAMES.get(
                int(label)
            )

            if class_name is None:
                continue

            x1, y1, x2, y2 = [
                float(value)
                for value in box
            ]

            # Calculate the centre from the model box.
            u = (
                        x1 + x2
                ) / 2.0

            v = (
                        y1 + y2
                ) / 2.0

            # Keep the same workspace check for every detector.
            valid, _reason = (
                passes_workspace_gate(
                    u,
                    v,
                )
            )

            if not valid:
                continue

            bbox_x = int(
                round(x1)
            )

            bbox_y = int(
                round(y1)
            )

            bbox_w = max(
                1,
                int(
                    round(
                        x2 - x1
                    )
                ),
            )

            bbox_h = max(
                1,
                int(
                    round(
                        y2 - y1
                    )
                ),
            )

            # Learned detectors use bounding box area.
            area = float(
                (x2 - x1)
                * (y2 - y1)
            )

            detections.append(
                Detection(
                    method=self.name,
                    class_name=class_name,
                    confidence=float(score),
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    u=float(u),
                    v=float(v),
                    area=area,
                )
            )

        # Keep the same ordering used by OpenCV.
        detections.sort(
            key=lambda detection: detection.area,
            reverse=True,
        )

        inference_time_ms = (
                                    time.perf_counter() - start
                            ) * 1000.0

        return detections, inference_time_ms
