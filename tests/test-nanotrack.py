# test_nanotrack.py

import pytest
from nanotrack import NanoTrack
import numpy as np

def test_nanotrack_initialization():
    tracker = NanoTrack()
    assert tracker.iou_threshold == 0.5
    assert tracker.max_age == 5
    assert tracker.min_hits == 3
    assert tracker.frame_count == 0
    assert tracker.track_id_count == 0
    assert len(tracker.tracks) == 0

def test_nanotrack_update_with_no_detections():
    tracker = NanoTrack()
    tracks = tracker.update([])
    assert tracks == []

def test_nanotrack_update_with_single_detection():
    tracker = NanoTrack()
    detection = np.array([100, 100, 200, 200])
    tracks = tracker.update([detection])
    assert len(tracks) == 0  # Should be 0 since min_hits is 3
    assert len(tracker.tracks) == 1

def test_nanotrack_iou():
    boxA = np.array([100, 100, 200, 200])
    boxB = np.array([150, 150, 250, 250])
    iou = NanoTrack.iou(boxA, boxB)
    assert 0.14 < iou < 0.16  # Expected IOU is around 0.1429
