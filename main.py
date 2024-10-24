import cv2
import numpy as np
from nanotrack import YOLOv8Detector, NanoTrack
import torch

def main():
    # Initialize detector and tracker
    detector = YOLOv8Detector(
        model_path="yolov8n.pt",  # or your custom model path

    )

    tracker = NanoTrack(
    )

    # Open video capture from a file
    video_path = r"C:\Users\tragu\Pictures\video.mp4"  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect(frame)

        # Update tracks
        tracks = tracker.update(detections)

        # Draw bounding boxes and track IDs
        for track in tracks:
            x1, y1, x2, y2, conf, class_id = track[:6]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('NanoTrack', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()