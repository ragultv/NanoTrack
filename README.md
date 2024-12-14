# **NanoTrack**

![Downloads](https://static.pepy.tech/badge/nanotrack)

**NanoTrack** is a lightweight and efficient object detection and tracking library designed for seamless integration with **YOLOv5** and **YOLOv8** models. It delivers **real-time tracking** with minimal resource usage, making it ideal for edge devices and systems with limited performance.

---

## **Features**
- 🚀 **Lightweight**: Optimized for minimal computational overhead.  
- 🎯 **Seamless Integration**: Fully compatible with YOLOv5 and YOLOv8.  
- ⚡ **Real-Time Performance**: Fast and accurate tracking for video streams.  
- 🛠️ **Simple API**: Easy-to-use interfaces for rapid development.  
- 📹 **Video & Stream Support**: Works with video files and live camera streams.

---

## **Installation**

### **Install via PyPI**
To install NanoTrack from PyPI, run:
```bash
pip install nanotrack
```

### **Install from GitHub**
For the latest version directly from the source:
```bash
pip install git+https://github.com/ragultv/nanotrack.git
```

---

## **Usage**

### **1. Import and Initialize**
```python
from nanotrack import YOLOv8Detector, NanoTrack
import cv2

# Initialize the YOLOv8 Detector
detector = YOLOv8Detector(model_path="yolov8n.pt")  # Replace with your model path

# Initialize the NanoTrack Tracker
tracker = NanoTrack()
```

### **2. Process Video for Detection and Tracking**
```python
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
```

---

## **Supported Models**
NanoTrack seamlessly works with:
- **YOLOv5**: Optimized and reliable object detection.  
- **YOLOv8**: Cutting-edge detection accuracy and performance.

---

## **Contributing**
We welcome contributions to NanoTrack!  
To contribute:  
1. **Fork** the repository.  
2. **Create a branch**:  
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and test thoroughly.  
4. **Submit a Pull Request** with a clear description.  

---

## **License**
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## **Support**
For issues, feature requests, or questions, feel free to:  
- Open an issue on our [GitHub repository](https://github.com/ragultv/nanotrack).  
- Reach out with feedback or suggestions.

---

### **Let’s Track Smarter, Faster, and Lighter with NanoTrack!** 🚀

---

