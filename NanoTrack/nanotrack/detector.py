import torch
from typing import List, Tuple
import numpy as np


class YOLOv5Detector:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.to(self.device)
        model.eval()
        return model

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> List[np.ndarray]:
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()

        filtered_detections = []
        for det in detections:
            if det[4] >= conf_threshold:
                filtered_detections.append(det)

        return np.array(filtered_detections)

    @staticmethod
    def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        indices = []
        sorted_indices = np.argsort(scores)[::-1]

        while len(sorted_indices) > 0:
            index = sorted_indices[0]
            indices.append(index)

            iou = YOLOv5Detector.iou(boxes[index], boxes[sorted_indices[1:]])
            sorted_indices = sorted_indices[1:][iou <= iou_threshold]

        return indices

    @staticmethod
    def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        left_top = np.maximum(box[:2], boxes[:, :2])
        right_bottom = np.minimum(box[2:], boxes[:, 2:])

        wh = np.maximum(right_bottom - left_top, 0)
        inter_area = wh[:, 0] * wh[:, 1]

        iou = inter_area / (box_area + boxes_area - inter_area + 1e-6)
        return iou