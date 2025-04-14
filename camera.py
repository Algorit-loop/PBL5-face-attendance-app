import cv2
import time
import threading
import numpy as np
import os
from typing import Dict, Any, Generator
from ultralytics import YOLO  # Sử dụng YOLOv8
import onnxruntime
import asyncio
from employeecontroller import EmployeeController  # Changed import
import joblib

class FaceScanner:
    def __init__(self):
        self.scanning = False
        self.frames_captured = 0
        self.max_frames = 50  # Thay đổi thành 50 frames
        self.employee_id = None
        self.scan_complete = False
        self.warning_message = None
        self.start_time = None
        
    def start_scan(self, employee_id):
        self.scanning = True
        self.employee_id = employee_id
        self.frames_captured = 0
        self.scan_complete = False
        self.warning_message = None
        self.start_time = time.time()
        os.makedirs(os.path.join("face_data", str(employee_id)), exist_ok=True)
        
    def capture_frame(self, frame):
        if not self.scanning:
            return False
            
        # Check if 2 seconds have passed since start
        if time.time() - self.start_time < 3:
            return False
            
        if self.frames_captured < self.max_frames:
            frame_path = os.path.join(
                "face_data", 
                str(self.employee_id), 
                f"frame_{self.frames_captured + 1:03d}.jpg"
            )
            if cv2.imwrite(frame_path, frame):
                self.frames_captured += 1
                
                # Check if scanning is complete
                if self.frames_captured >= self.max_frames:
                    self.scanning = False
                    self.scan_complete = True
                return True
            return False
        return False
    
    def set_warning(self, message):
        self.warning_message = message
            
    def get_status(self):
        return {
            "scanning": self.scanning,
            "frames_captured": self.frames_captured,
            "max_frames": self.max_frames,
            "scan_complete": self.scan_complete,
            "warning_message": self.warning_message,
            "progress": (self.frames_captured / self.max_frames) * 100
        }

# Load YOLOv8 face model (sử dụng phiên bản nhẹ hơn)
model = YOLO("yolov8n-face-lindevs.pt")

# Khởi tạo onnxruntime session cho model R50
onnx_session = onnxruntime.InferenceSession("R50.onnx")

# Initialize face scanner
face_scanner = FaceScanner()

def process_with_R50(face_image: np.ndarray) -> np.ndarray:
    """
    Tiền xử lý ảnh khuôn mặt, resize về kích thước 112x112 theo yêu cầu của model R50,
    chuyển sang RGB, chuẩn hóa (trừ 127.5, chia 128.0), và chuyển về định dạng NCHW.
    Sau đó chạy inference và trả về vector embedding.
    
    Args:
        face_image (np.ndarray): Ảnh khuôn mặt đầu vào (BGR, HWC format).
        
    Returns:
        np.ndarray: Vector embedding (shape (512,) hoặc (1, 512)).
    """
    try:
        # Resize ảnh khuôn mặt về 112x112
        face_resized = cv2.resize(face_image, (112, 112))
        
        # Chuyển sang RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Chuyển sang float32 và chuẩn hóa theo ArcFace
        face_float = face_rgb.astype(np.float32)
        face_float = (face_float - 127.5) / 128.0
        
        # Chuyển sang định dạng NCHW (1, 3, 112, 112)
        face_input = np.transpose(face_float, (2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)
        
        # Chạy inference với ONNX model
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: face_input})
        embedding = outputs[0]
        
        # Chuẩn hóa embedding để tương thích với SVM
        if embedding.ndim > 2:  # Trường hợp (1, 1, 512)
            embedding = embedding.reshape(-1)
        elif embedding.ndim == 2:  # Trường hợp (1, 512)
            embedding = embedding.reshape(-1)  # Trả về (512,)
        
        print(f"Embedding shape from R50: {embedding.shape}")
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

def load_models():
    global svm_model, label_encoder
    try:
        svm_model = joblib.load("svm_face_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        print("Reloaded SVM model and LabelEncoder successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

# Tải mô hình SVM và LabelEncoder
load_models()

# Global dictionary for employee mapping
employee_dict = {}

def update_employee_mapping():
    """Update the employee ID to name mapping"""
    global employee_dict
    try:
        # Get employees using EmployeeController
        employees = asyncio.run(EmployeeController.get_all())
        employee_dict = {str(emp["id"]): emp["full_name"] for emp in employees}
        print(f"Updated employee mapping with {len(employee_dict)} employees")
    except Exception as e:
        print(f"Error updating employee mapping: {str(e)}")

# Initial update of employee mapping
update_employee_mapping()
print(employee_dict)

class Camera:
    @classmethod
    def reload_model(cls):
        """Reload the face recognition model"""
        load_models()
        update_employee_mapping()

def yolo_r50_inference(original_frame: np.ndarray, yolo_frame: np.ndarray) -> None:
    """
    Hàm chạy inference của YOLO trên yolo_frame (ví dụ 640x640) và sau đó thực hiện inference
    model R50 trên mỗi khuôn mặt được phát hiện, dùng hệ số scale để chuyển tọa độ sang original_frame.
    Dự đoán id_user bằng SVM với ngưỡng xác suất 0.7, lưu vào kết quả.
    """
    global global_result, processing
    try:
        start_time = time.time()  # Ghi lại thời gian bắt đầu
        
        # Chạy inference YOLO trên yolo_frame
        res = model(yolo_frame, verbose=True)[0]
        
        # Tạo danh sách để lưu kết quả tùy chỉnh
        custom_results = []
        
        # Tính scaling factors từ khung hình YOLO đến original_frame
        scale_x = original_frame.shape[1] / yolo_frame.shape[1]
        scale_y = original_frame.shape[0] / yolo_frame.shape[0]
        
        # Duyệt qua từng bounding box
        for box in res.boxes:
            # Lấy tọa độ (trên khung hình yolo_frame)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Chuyển tọa độ sang khung hình original_frame
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            # Crop khuôn mặt từ original_frame
            face_crop = original_frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
            if face_crop.size > 0:
                # Trích xuất embedding bằng R50
                embedding = process_with_R50(face_crop)
                print("Embedding vector từ model R50:", embedding.shape)
                
                # Chuẩn hóa embedding thành shape (1, 512)
                if embedding.ndim == 3:  # Trường hợp (1, 1, 512)
                    embedding = embedding.reshape(1, -1)
                elif embedding.ndim == 2:  # Trường hợp (1, 512)
                    embedding = embedding.reshape(1, -1)
                else:  # Trường hợp (512,)
                    embedding = embedding.reshape(1, -1)
                
                # Dự đoán id_user bằng SVM với xác suất
                try:
                    prob = svm_model.predict_proba(embedding)[0]
                    max_prob = np.max(prob)
                    print(f"Max probability: {max_prob:.2f}")
                    if max_prob >= 0.6: # Ngưỡng xác suất
                        pred = svm_model.predict(embedding)[0]
                        id_user = label_encoder.inverse_transform([pred])[0]
                        # Use the employee dictionary to get the name
                        print(f"Predicted id_user: {id_user}")
                        name = employee_dict.get(str(id_user), "Unknown")
                    else:
                        id_user = "Unknown"
                        name = "Unknown"
                except Exception as e:
                    print(f"Lỗi dự đoán SVM: {str(e)}")
                    id_user = "Unknown"
                    name = "Unknown"
                
                # Lưu thông tin box, nhãn, và name
                custom_results.append({
                    'xyxy': [x1, y1, x2, y2],
                    'conf': conf,
                    'id_user': id_user,
                    'name': name
                })
            else:
                # Nếu không crop được khuôn mặt
                custom_results.append({
                    'xyxy': [x1, y1, x2, y2],
                    'conf': conf,
                    'id_user': "Unknown",
                    'name': "Unknown"
                })
        
        # Lưu kết quả tùy chỉnh vào global_result
        global_result = custom_results
        
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
            print(f"Độ trễ: {(current_time - last_frame_time)*1000:.3f} ms")
            if camera_running and camera is not None and camera.isOpened():
                try:
                    success, frame = camera.read()
                    if not success:
                        frame = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        # Lật ảnh và resize
                        frame = cv2.flip(frame, 1)
                        frame = cv2.resize(frame, (width, height))
                        
                        # Tạo frame cho YOLO
                        yolo_frame = cv2.resize(frame, (640, 640))
                        
                        frame_count += 1
                        # Chạy inference nếu không có thread nào đang xử lý
                        if frame_count % 1 == 0 and not processing:
                            processing = True
                            thread = threading.Thread(target=yolo_r50_inference,
                                                    args=(frame.copy(), yolo_frame.copy()))
                            thread.daemon = True
                            thread.start()
                        
                        # Vẽ bounding box và nhãn từ global_result
                        if global_result is not None:
                            scale_x = width / 640
                            scale_y = height / 640
                            for result in global_result:
                                x1, y1, x2, y2 = map(int, result['xyxy'])
                                conf = result['conf']
                                name = result['name']
                                
                                # Chuyển tọa độ sang khung hình gốc
                                x1_scaled = int(x1 * scale_x)
                                y1_scaled = int(y1 * scale_y)
                                x2_scaled = int(x2 * scale_x)
                                y2_scaled = int(y2 * scale_y)
                                
                                # Vẽ bounding box
                                cv2.rectangle(frame, (x1_scaled, y1_scaled), 
                                            (x2_scaled, y2_scaled), (0, 255, 0), 2)
                                
                                # Vẽ conf phía trên bounding box
                                conf_label = f"{conf:.2f}"
                                cv2.putText(frame, conf_label, (x1_scaled, y1_scaled - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # Vẽ Name phía dưới bounding box
                                name_label = f"Name: {name}"
                                cv2.putText(frame, name_label, (x1_scaled, y2_scaled + 20),
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

def scan_frames() -> Generator[bytes, None, None]:
    """
    Similar to generate_frames but specifically for scanning faces during employee registration.
    Only uses YOLOv8 for face detection and saves detected faces to face_data directory.
    """
    global last_frame_time, frame_count, processing, global_result, face_scanner
    
    while True:
        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            if camera_running and camera is not None and camera.isOpened():
                try:
                    success, frame = camera.read()
                    if not success:
                        frame = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        frame = cv2.flip(frame, 1)
                        frame = cv2.resize(frame, (width, height))
                        
                        # Create YOLO input frame
                        yolo_frame = cv2.resize(frame, (640, 640))
                        
                        # Run YOLO inference for face detection
                        if frame_count % 1 == 0 and not processing:
                            processing = True
                            results = model(yolo_frame, verbose=True)[0]
                            processing = False
                            
                            # Check number of faces detected
                            num_faces = len(results.boxes)
                            
                            # If scanning is active, handle face detection results
                            if face_scanner.scanning:
                                if num_faces == 0:
                                    face_scanner.set_warning("Không tìm thấy khuôn mặt")
                                elif num_faces > 1:
                                    face_scanner.set_warning("Phát hiện nhiều khuôn mặt! Vui lòng chỉ để một người trong khung hình")
                                    
                                else:
                                    face_scanner.set_warning(None)
                                    # Get the single face detected
                                    face_box = results.boxes[0]
                                    x1, y1, x2, y2 = map(int, face_box.xyxy[0])
                                    
                                    # Scale coordinates to original frame size
                                    scale_x = width / 640
                                    scale_y = height / 640
                                    x1_scaled = int(x1 * scale_x)
                                    y1_scaled = int(y1 * scale_y)
                                    x2_scaled = int(x2 * scale_x)
                                    y2_scaled = int(y2 * scale_y)
                                    
                                    # Extract and save face only if one face is detected
                                    face_img = frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
                                    if face_img.size > 0:
                                        face_scanner.capture_frame(face_img)
                            
                            # Draw face rectangles and status
                            for box in results.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                
                                # Scale coordinates
                                x1_scaled = int(x1 * scale_x)
                                y1_scaled = int(y1 * scale_y)
                                x2_scaled = int(x2 * scale_x)
                                y2_scaled = int(y2 * scale_y)
                                
                                # Use red color for multiple faces, green for single face
                                color = (0, 0, 255) if num_faces > 1 else (0, 255, 0)
                                
                                # Draw rectangle
                                cv2.rectangle(frame, (x1_scaled, y1_scaled), 
                                           (x2_scaled, y2_scaled), color, 2)
                                
                                # Add confidence label
                                label = f"Face {conf:.2f}"
                                cv2.putText(frame, label, (x1_scaled, y1_scaled - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        frame_count += 1
                        
                        # Add scanning overlay
                        if face_scanner.scanning:
                            # Draw frame count for current direction
                            cv2.putText(frame, 
                                      f"Frames: {face_scanner.frames_captured}/{face_scanner.max_frames}",
                                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Draw warning message if exists
                            if face_scanner.warning_message:
                                cv2.putText(frame, face_scanner.warning_message,
                                          (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.7, (0, 0, 255), 2)
                            
                except Exception as e:
                    print(f"Lỗi khi xử lý frame: {str(e)}")
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            last_frame_time = current_time
        else:
            time.sleep(0.001)
