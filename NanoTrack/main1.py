from nanotrack import YOLOv8Detector, NanoTrack
import cv2

# Initialize the YOLOv8 Detector
detector = YOLOv8Detector(model_path="yolov8n.pt")  # Replace with your model path

# Initialize the NanoTrack Tracker
tracker = NanoTrack()
# Load video file or webcam input
cap = cv2.VideoCapture("path_to_video.mp4")  # Replace with your video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    detections = detector.detect(frame)

    # Update tracker with detections
    tracks = tracker.update(detections)

    # Draw bounding boxes and track IDs
    for track in tracks:
        x1, y1, x2, y2, _, _, track_id = track[:7]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Display the results
    cv2.imshow("NanoTrack", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
