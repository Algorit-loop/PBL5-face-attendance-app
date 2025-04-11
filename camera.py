import cv2
import time
import threading
import numpy as np
from typing import Dict, Any, Generator
from ultralytics import YOLO  # Sử dụng YOLOv8
import onnxruntime

# Load YOLOv8 face model (sử dụng phiên bản nhẹ hơn)
model = YOLO("yolov8n-face-lindevs.pt")

# Khởi tạo onnxruntime session cho model R50
onnx_session = onnxruntime.InferenceSession("R18.onnx")

def process_with_R50(face_image: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh khuôn mặt, resize về kích thước 112x112 theo yêu cầu của model R50,
    chuyển sang RGB, chuẩn hóa và chuyển về định dạng NCHW. Sau đó chạy inference và
    trả về vector embedding.
    """
    try:
        # Resize ảnh khuôn mặt về 112x112
        face_resized = cv2.resize(face_image, (112, 112))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_float = face_rgb.astype(np.float32) / 255.0
        
        # Nếu cần thêm bước chuẩn hóa (trừ mean, chia std) thì chèn ở đây.
        # face_float = (face_float - mean) / std

        # Chuyển sang định dạng NCHW
        face_input = np.transpose(face_float, (2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)  # Kết quả: (1, 3, 112, 112)

        # Lấy tên input từ model (giả sử model có 1 input)
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: face_input})
        embedding = outputs[0]
        return embedding
    except Exception as e:
        print(f"Lỗi khi xử lý với model R50: {str(e)}")
        return np.array([])

width = 960
height = 720

# Camera setup
camera = None
camera_running = False
last_frame_time = 0
frame_interval = 1/60  # mục tiêu 60 FPS

# Global variables cho xử lý bất đồng bộ
global_result = None  # Kết quả YOLO (và sau đó có thể gồm thêm thông tin của R50)
processing = False    # Cờ báo trạng thái inference đang chạy hay không
frame_count = 0       # Đếm số frame

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

def yolo_r50_inference(original_frame: np.ndarray, yolo_frame: np.ndarray) -> None:
    """
    Hàm chạy inference của YOLO trên yolo_frame (ví dụ 640x640) và sau đó thực hiện inference
    model R50 trên mỗi khuôn mặt được phát hiện, dùng hệ số scale để chuyển tọa độ sang original_frame.
    Kết quả (bounding box và vector embedding) sẽ được xử lý trong hàm này.
    """
    global global_result, processing
    try:
        start_time = time.time()  # Ghi lại thời gian bắt đầu
        
        # Chạy inference YOLO trên yolo_frame
        res = model(yolo_frame, verbose=True)[0]
        global_result = res  # Lưu kết quả YOLO
        
        # Tính scaling factors từ khung hình YOLO đến original_frame
        scale_x = original_frame.shape[1] / yolo_frame.shape[1]
        scale_y = original_frame.shape[0] / yolo_frame.shape[0]
        
        # Duyệt qua từng bounding box
        for box in res.boxes:
            # Lấy tọa độ (trên khung hình yolo_frame)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Face {conf:.2f}"
            
            # Chuyển tọa độ sang khung hình original_frame
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            # Crop khuôn mặt từ original_frame (cần kiểm tra vùng crop hợp lệ)
            face_crop = original_frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
            if face_crop.size > 0:
                embedding = process_with_R50(face_crop)
                # print("Embedding vector từ model R50:", embedding.shape)
        
        end_time = time.time()  # Ghi lại thời gian kết thúc
        print(f"Thời gian chạy yolo_r50_inference: {(end_time - start_time) * 1000:.2f} ms")
    except Exception as e:
        print(f"Lỗi khi chạy inference kết hợp YOLO và R50: {str(e)}")
    processing = False

def generate_frames() -> Generator[bytes, None, None]:
    global last_frame_time, frame_count, processing, global_result
    while True:
        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            # Đo độ trễ
            print(f"Độ trễ : {(current_time - last_frame_time)*1000:.3f} ms")
            if camera_running and camera is not None and camera.isOpened():
                try:
                    success, frame = camera.read()
                    if not success:
                        frame = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        # Lật ảnh để tạo hiệu ứng gương và resize về kích thước gốc
                        frame = cv2.flip(frame, 1)
                        frame = cv2.resize(frame, (width, height))
                        
                        # Tạo frame dùng cho YOLO: resize về 640x640
                        yolo_frame = cv2.resize(frame, (640, 640))
                        
                        frame_count += 1
                        # Nếu không có inference nào đang chạy, khởi tạo một thread inference chạy cả YOLO và R50
                        if frame_count % 1 == 0 and not processing:
                            processing = True
                            # Ở đây ta truyền cả frame gốc và frame cho YOLO
                            thread = threading.Thread(target=yolo_r50_inference,
                                                      args=(frame.copy(), yolo_frame.copy()))
                            thread.daemon = True
                            thread.start()
                        
                        # Vẽ các bounding box dựa trên kết quả của YOLO (nếu đã có)
                        if global_result is not None:
                            scale_x = width / 640
                            scale_y = height / 640
                            for box in global_result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                label = f"Face {conf:.2f}"
                                x1_scaled = int(x1 * scale_x)
                                y1_scaled = int(y1 * scale_y)
                                x2_scaled = int(x2 * scale_x)
                                y2_scaled = int(y2 * scale_y)
                                cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled),
                                              (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1_scaled, y1_scaled - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Lỗi khi xử lý frame: {str(e)}")
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Mã hóa frame thành JPEG để stream hoặc hiển thị
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            last_frame_time = current_time
        else:
            time.sleep(0.001)
