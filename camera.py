import cv2
import time
import threading
import numpy as np
from typing import Dict, Any, Generator
from ultralytics import YOLO  # Dùng YOLOv8

# Load YOLOv8 face model (sử dụng phiên bản n nhẹ hơn)
model = YOLO("yolov8n-face-lindevs.pt")

width = 960
height = 720

# Camera setup
camera = None
camera_running = False
last_frame_time = 0
frame_interval = 1/60  # cho khoảng thời gian giữa các frame (mục tiêu 60 FPS)

# Global variables cho xử lý bất đồng bộ
result = None         # Kết quả được sử dụng để vẽ khung khuôn mặt
cur_result = None     # Kết quả tạm thời sau mỗi lần inference hoàn tất
processing = False    # Cờ báo trạng thái inference đang chạy hay không
frame_count = 0       # Đếm số frame

def init_camera() -> bool:
    global camera, camera_running
    try:
        if camera is None:
            # Nếu có thể, bạn có thể thử backend khác như:
            # camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

def model_inference(frame_for_detection: np.ndarray) -> None:
    global result, cur_result, processing
    try:
        # Chạy model trên frame đã được resize về kích thước (640, 640)
        res = model(frame_for_detection, verbose=False)[0]
        cur_result = res  # cập nhật kết quả tạm thời
        result = cur_result  # gán result = cur_result sau khi inference xong
    except Exception as e:
        print(f"Lỗi khi chạy inference: {str(e)}")
    processing = False

def generate_frames() -> Generator[bytes, None, None]:
    global last_frame_time, frame_count, processing, result
    while True:
        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            print(f"Độ trễ : {(current_time - last_frame_time)*1000:.3f} ms")
            if camera_running and camera is not None and camera.isOpened():
                try:
                    success, frame = camera.read()
                    if not success:
                        frame = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        # Lật ảnh để tạo hiệu ứng gương
                        frame = cv2.flip(frame, 1)
                        # Resize frame về kích thước mong muốn
                        frame = cv2.resize(frame, (width, height))
                        
                        # Tạo bản sao frame dùng cho model (resize về 640x640)
                        resized_frame = cv2.resize(frame, (640, 640))
                        
                        # Tăng biến đếm frame
                        frame_count += 1
                        # Nếu là frame thứ 3 (mỗi 3 frame) và không có inference nào đang chạy
                        if frame_count % 1 == 0 and not processing:
                            processing = True
                            # Sử dụng luồng riêng để thực hiện inference bất đồng bộ
                            thread = threading.Thread(target=model_inference, args=(resized_frame.copy(),))
                            thread.daemon = True
                            thread.start()
                        
                        # Nếu có result (từ inference trước đó), vẽ khung khuôn mặt lên frame
                        if result is not None:
                            # Tính scaling factors từ 640x640 về kích thước gốc (width x height)
                            scale_x = width / 640
                            scale_y = height / 640
                            # Vẽ khung khuôn mặt từ kết quả của YOLO
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                label = f"Face {conf:.2f}"
        
                                # Scale to target resolution
                                x1 = int(x1 * scale_x)
                                y1 = int(y1 * scale_y)
                                x2 = int(x2 * scale_x)
                                y2 = int(y2 * scale_y)
        
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Lỗi khi xử lý frame: {str(e)}")
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Mã hóa frame thành JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            last_frame_time = current_time
        else:
            time.sleep(0.001)
