import numpy as np
from typing import List
from scipy.optimize import linear_sum_assignment

class NanoTrack:
    def __init__(self, iou_threshold: float = 0.5, max_age: int = 5, min_hits: int = 3):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
        self.track_id_count = 0

    def update(self, detections: List[np.ndarray]) -> List[np.ndarray]:
        self.frame_count += 1

        # Predict new locations of existing tracks
        for track in self.tracks:
            self._predict(track)

        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(detections)

        # Debug: Print matched indices
        #print("Matched indices:", matched)
        #print("Unmatched detections:", unmatched_dets)
        #print("Unmatched tracks:", unmatched_trks)

        # Update matched tracks
        for trk_idx, det_idx in matched:
            if trk_idx < len(self.tracks) and det_idx < len(detections):
                self._update_track(self.tracks[trk_idx], detections[det_idx])
            #else:
            #print(f"Skipping update for trk_idx: {trk_idx}, det_idx: {det_idx}")

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._initiate_track(detections[det_idx])

        # Remove dead tracks
        self.tracks = [trk for trk in self.tracks if trk['time_since_update'] < self.max_age]

        return [trk['bbox'] for trk in self.tracks if trk['hits'] >= self.min_hits]

    def _predict(self, track):
        if track['age'] > 0:
            track['bbox'][:4] += track['velocity']
        track['age'] += 1
        track['time_since_update'] += 1

    def _associate_detections_to_tracks(self, detections):
        if len(self.tracks) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.tracks):
                iou_matrix[d, t] = self.iou(det, trk['bbox'])

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.asarray(matched_indices).T

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_tracks = []
        for t, trk in enumerate(self.tracks):
            if t not in matched_indices[:, 1]:
                unmatched_tracks.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_tracks.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_detections, unmatched_tracks

    def _update_track(self, track, detection):
        track['bbox'] = detection
        track['hits'] += 1
        track['time_since_update'] = 0
        if track['age'] > 1:
            track['velocity'] = detection[:4] - track['bbox'][:4]

    def _initiate_track(self, detection):
        self.track_id_count += 1
        self.tracks.append({
            'bbox': detection,
            'id': self.track_id_count,
            'hits': 1,
            'age': 1,
            'time_since_update': 0,
            'velocity': np.zeros(4)
        })

    @staticmethod
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxB[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
