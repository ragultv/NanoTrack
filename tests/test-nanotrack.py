import pytest
import numpy as np
from nanotrack.nanotrack import NanoTrack


def test_nanotrack_initialization():
    tracker = NanoTrack(iou_threshold=0.5, max_age=5, min_hits=3)
    assert tracker.iou_threshold == 0.5
    assert tracker.max_age == 5
    assert tracker.min_hits == 3
    assert tracker.frame_count == 0
    assert len(tracker.tracks) == 0


def test_nanotrack_single_detection():
    tracker = NanoTrack()
    detections = [np.array([100, 100, 200, 200, 0.9])]
    tracks = tracker.update(detections)
    assert len(tracks) == 1
    assert np.allclose(tracks[0][:4], [100, 100, 200, 200])


def test_nanotrack_multiple_detections():
    tracker = NanoTrack()
    detections = [
        np.array([100, 100, 200, 200, 0.9]),
        np.array([300, 300, 400, 400, 0.85])
    ]
    tracks = tracker.update(detections)
    assert len(tracks) == 2
    assert np.allclose(tracks[0][:4], [100, 100, 200, 200])
    assert np.allclose(tracks[1][:4], [300, 300, 400, 400])


def test_nanotrack_track_aging_and_deletion():
    tracker = NanoTrack(max_age=2)
    detections = [np.array([100, 100, 200, 200, 0.9])]
    tracker.update(detections)

    # No detections for the next frames, tracks should be deleted after max_age
    for _ in range(2):
        tracks = tracker.update([])

    assert len(tracks) == 0


def test_nanotrack_matching_detections_to_tracks():
    tracker = NanoTrack(iou_threshold=0.3)
    detections = [
        np.array([100, 100, 200, 200, 0.9]),
        np.array([300, 300, 400, 400, 0.85])
    ]
    tracks = tracker.update(detections)

    # Next frame with slightly moved detections
    new_detections = [
        np.array([105, 105, 205, 205, 0.92]),  # should match first track
        np.array([305, 305, 405, 405, 0.88])  # should match second track
    ]
    tracks = tracker.update(new_detections)

    assert len(tracks) == 2
    assert np.allclose(tracks[0][:4], [105, 105, 205, 205])
    assert np.allclose(tracks[1][:4], [305, 305, 405, 405])


def test_nanotrack_empty_detections():
    tracker = NanoTrack()
    tracks = tracker.update([])
    assert len(tracks) == 0


def test_nanotrack_out_of_bound_values():
    tracker = NanoTrack()
    detections = [
        np.array([-10, -10, 100, 100, 0.9]),  # partially out of bounds
        np.array([900, 900, 1000, 1000, 0.85])  # completely out of bounds
    ]
    tracks = tracker.update(detections)
    assert len(tracks) == 1  # Only one valid track should be created
    assert np.allclose(tracks[0][:4], [-10, -10, 100, 100])


def test_nanotrack_complex_scenario():
    tracker = NanoTrack(iou_threshold=0.3)
    # Frame 1: Two detections
    detections1 = [
        np.array([100, 100, 200, 200, 0.9]),
        np.array([300, 300, 400, 400, 0.85])
    ]
    tracker.update(detections1)

    # Frame 2: Slightly moved detections, should match
    detections2 = [
        np.array([110, 110, 210, 210, 0.92]),
        np.array([310, 310, 410, 410, 0.88])
    ]
    tracks = tracker.update(detections2)
    assert len(tracks) == 2

    # Frame 3: One detection moves out, another one moves in, and a new detection appears
    detections3 = [
        np.array([120, 120, 220, 220, 0.95]),  # Continues from previous track
        np.array([500, 500, 600, 600, 0.93])  # New detection
    ]
    tracks = tracker.update(detections3)
    assert len(tracks) == 3
