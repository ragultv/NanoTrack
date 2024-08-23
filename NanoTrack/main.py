import cv2
import numpy as np
import cv2
import numpy as np
from nanotrack.nanotrack import NanoTrack
from nanotrack.detector import YOLOv5Detector

def main():
    # Initialize detector and tracker
    detector = YOLOv5Detector(model_path=r"C:\Users\tragu\Downloads\yolov5s.pt")
    tracker = NanoTrack()

    # Open video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

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
            cv2.putText(frame, f"ID: {int(track[-1])}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('NanoTrack', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()