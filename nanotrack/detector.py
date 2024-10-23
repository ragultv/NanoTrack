from abc import ABC, abstractmethod
import torch
from typing import List, Tuple, Optional, Union
import numpy as np
from ultralytics import YOLO


class BaseDetector(ABC):
    """Base class for YOLO detectors"""

    def __init__(self, model_path: str, device: str = 'cpu', conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize base detector

        Args:
            model_path: Path to model weights
            device: Device to run inference on ('cpu' or 'cuda')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model(model_path)

    @abstractmethod
    def _load_model(self, model_path: str):
        """Load model - to be implemented by subclasses"""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect objects in image

        Args:
            image: Input image as numpy array (H,W,C) in BGR format

        Returns:
            Array of detections in format [x1,y1,x2,y2,conf,class_id]
        """
        pass

    @staticmethod
    def clip_boxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Clip boxes to image boundaries"""
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_shape[0])
        return boxes


class YOLOv5Detector(BaseDetector):
    """YOLOv5 detector implementation"""

    def _load_model(self, model_path: str) -> YOLO:
        """Load YOLOv8 model"""
        model = YOLO(model_path)
        model.to(self.device)
        return model

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect objects using YOLOv5

        Args:
            image: Input image as numpy array (H,W,C) in BGR format

        Returns:
            Array of detections in format [x1,y1,x2,y2,conf,class_id]
        """
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()

        # Filter by confidence
        mask = detections[:, 4] >= self.conf_threshold
        detections = detections[mask]

        # Apply NMS if there are detections
        if len(detections) > 0:
            keep = self.non_max_suppression(
                detections[:, :4],
                detections[:, 4],
                self.iou_threshold
            )
            detections = detections[keep]

        return detections

    @staticmethod
    def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Apply Non-Maximum Suppression"""
        indices = []
        sorted_indices = np.argsort(scores)[::-1]

        while len(sorted_indices) > 0:
            index = sorted_indices[0]
            indices.append(index)

            if len(sorted_indices) == 1:
                break

            iou = YOLOv5Detector.iou(boxes[index], boxes[sorted_indices[1:]])
            sorted_indices = sorted_indices[1:][iou <= iou_threshold]

        return indices

    @staticmethod
    def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between box and boxes"""
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        left_top = np.maximum(box[:2], boxes[:, :2])
        right_bottom = np.minimum(box[2:], boxes[:, 2:])

        wh = np.maximum(right_bottom - left_top, 0)
        inter_area = wh[:, 0] * wh[:, 1]

        iou = inter_area / (box_area + boxes_area - inter_area + 1e-6)
        return iou


class YOLOv8Detector(BaseDetector):
    """YOLOv8 detector implementation"""

    def _load_model(self, model_path: str) -> YOLO:
        """Load YOLOv8 model"""
        model = YOLO(model_path)
        model.to(self.device)
        return model

    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect objects using YOLOv8

        Args:
            image: Input image as numpy array (H,W,C) in BGR format

        Returns:
            Array of detections in format [x1,y1,x2,y2,conf,class_id]
        """
        results = self.model(image, verbose=False)[0]
        detections = []

        # Extract boxes, scores and class ids
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        # Filter by confidence
        mask = scores >= self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Combine into detection format
        for box, score, class_id in zip(boxes, scores, class_ids):
            detection = np.concatenate([box, [score], [class_id]])
            detections.append(detection)

        return np.array(detections) if detections else np.empty((0, 6))


def create_detector(
        model_path: str,
        version: str = 'v8',
        device: str = 'cpu',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
) -> Union[YOLOv5Detector, YOLOv8Detector]:
    """
    Factory function to create appropriate YOLO detector

    Args:
        model_path: Path to model weights
        version: YOLO version ('v5' or 'v8')
        device: Device to run inference on
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold

    Returns:
        YOLOv5Detector or YOLOv8Detector instance

    Raises:
        ValueError: If version is not 'v5' or 'v8'
    """
    if version.lower() == 'v5':
        return YOLOv5Detector(model_path, device, conf_threshold, iou_threshold)
    elif version.lower() == 'v8':
        return YOLOv8Detector(model_path, device, conf_threshold, iou_threshold)
    else:
        raise ValueError("Version must be 'v5' or 'v8'")