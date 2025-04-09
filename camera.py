import cv2
import time
import numpy as np
from typing import Dict, Any, Generator
from ultralytics import YOLO  # Thêm dòng này để dùng YOLOv8

# Load YOLOv8 face model
# model = YOLO("yolov8s-face-lindevs.pt") # yolov8n-face-lindevs.pt
model = YOLO("yolov8n-face-lindevs.pt")


# Camera setup
camera = None
camera_running = False
last_frame_time = 0
frame_interval = 1/30  # 30 FPS

def init_camera() -> bool:
    global camera, camera_running
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise Exception("Không thể mở camera")
            camera_running = True
            return True
    except Exception as e:
        print(f"Lỗi khi khởi tạo camera: {str(e)}")
        camera_running = False
        return False
    return True

def release_camera() -> None:
    global camera, camera_running
    try:
        if camera is not None:
            camera.release()
            camera = None
        camera_running = False
    except Exception as e:
        print(f"Lỗi khi giải phóng camera: {str(e)}")
        camera_running = False

def get_camera_info() -> Dict[str, Any]:
    if camera is None or not camera.isOpened():
        return {
            "width": 0,
            "height": 0,
            "fps": 0,
            "error": "Camera chưa được khởi tạo"
        }
    try:
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = camera.get(cv2.CAP_PROP_FPS)
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "error": None
        }
    except Exception as e:
        return {
            "width": 0,
            "height": 0,
            "fps": 0,
            "error": str(e)
        }

width = 1280
height = 720
def generate_frames() -> Generator[bytes, None, None]:
    global last_frame_time
    while True:
        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            if camera_running and camera is not None and camera.isOpened():
                try:
                    success, frame = camera.read()
                    if not success:
                        frame = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        # Flip frame horizontally for mirror effect
                        frame = cv2.flip(frame, 1)
                        
                        # Resize frame to target resolution
                        frame = cv2.resize(frame, (width, height))

                        # Resize frame về đúng input size cho YOLOv8 (640x640)
                        resized_frame = cv2.resize(frame, (640, 640))

                        # Thời gian bắt đầu
                        start_time = time.time()

                        # Chạy nhận diện
                        results = model(resized_frame, verbose=False)[0]

                        # Thời gian kết thúc
                        end_time = time.time()

                        # In thông tin
                        print(f"[INFO] Frame processed - {len(results.boxes)} face(s) detected - Time: {(end_time - start_time)*1000:.2f} ms")

                        # Calculate scaling factors from 640x640 to 1280x720
                        scale_x = width / 640
                        scale_y = height / 640

                        # Vẽ khung mặt người
                        for box in results.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            label = f"Face {conf:.2f}"

                            # Scale coordinates to our target resolution
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)

                            # Draw rectangle and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        
                except Exception as e:
                    print(f"Lỗi khi đọc frame: {str(e)}")
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            last_frame_time = current_time
        else:
            time.sleep(0.001)
