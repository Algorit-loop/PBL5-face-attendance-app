import cv2
import time
import threading
import numpy as np
import os
from typing import Dict, Any, Generator
from ultralytics import YOLO  # Sử dụng YOLOv8
import onnxruntime
import asyncio
from controllers.employee_controller import EmployeeController  # Changed import
import joblib
import serial  # For Arduino communication

class ServoController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):  # Adjust port as needed
        self.port = port
        self.baudrate = baudrate
        self.is_open = False
        self.setup_serial()
        
    def setup_serial(self):
        """Initialize serial communication with Arduino"""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print("Arduino serial connection initialized successfully")
        except Exception as e:
            print(f"Error initializing Arduino serial connection: {str(e)}")
            self.serial = None
            
    def open_door(self):
        """Open the door by sending 'o' command to Arduino"""
        try:
            if not self.is_open and self.serial:
                self.serial.write(b'o')  # Send 'o' command to open door
                time.sleep(1.5)  # Wait for servo to complete movement
                self.is_open = True
                print("Door opened")
        except Exception as e:
            print(f"Error opening door: {str(e)}")
            
    def close_door(self):
        """Close the door by sending 'c' command to Arduino"""
        try:
            if self.is_open and self.serial:
                self.serial.write(b'c')  # Send 'c' command to close door
                time.sleep(1.5)  # Wait for servo to complete movement
                self.is_open = False
                print("Door closed")
        except Exception as e:
            print(f"Error closing door: {str(e)}")
            
    def cleanup(self):
        """Clean up serial connection"""
        try:
            if self.serial:
                self.serial.close()
                print("Arduino serial connection closed")
        except Exception as e:
            print(f"Error closing Arduino serial connection: {str(e)}")

class FaceScanner:
    def __init__(self):
        self.scanning = False
        self.capturing = False  # New flag to indicate if we're capturing frames
        self.frames_captured = 0
        self.max_frames = 50  # Thay đổi thành 50 frames
        self.employee_id = None
        self.scan_complete = False
        self.warning_message = None
        self.start_time = None
        
    def start_scan(self, employee_id):
        self.scanning = True
        self.capturing = False  # Initialize as false - will be set to true with start_capture
        self.employee_id = employee_id
        self.frames_captured = 0
        self.scan_complete = False
        self.warning_message = None
        self.start_time = time.time()
        os.makedirs(os.path.join("face_data", str(employee_id)), exist_ok=True)
    
    def start_capture(self):
        """Start capturing frames after scanning has begun"""
        if self.scanning:
            self.capturing = True
            self.start_time = time.time()  # Reset start time when we begin capturing
            return True
        return False
        
    def capture_frame(self, frame):
        if not self.scanning or not self.capturing:
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
                    self.capturing = False
                    self.scan_complete = True
                return True
            return False
        return False
    
    def set_warning(self, message):
        self.warning_message = message
            
    def get_status(self):
        return {
            "scanning": self.scanning,
            "capturing": self.capturing,
            "frames_captured": self.frames_captured,
            "max_frames": self.max_frames,
            "scan_complete": self.scan_complete,
            "warning_message": self.warning_message,
            "progress": (self.frames_captured / self.max_frames) * 100
        }
        
    def stop(self):
        """Stop all scanning and capturing operations"""
        self.scanning = False
        self.capturing = False
        self.warning_message = None

# Load YOLOv8 face model (sử dụng phiên bản nhẹ hơn)
model = YOLO("yolov8n-face-lindevs.pt")

# Khởi tạo onnxruntime session cho model R50
onnx_session = onnxruntime.InferenceSession("R50.onnx")

# Initialize face scanner
face_scanner = FaceScanner()

# Initialize servo controller
servo_controller = ServoController(port='COM6')  # Change to your Arduino's port

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

# Stream URL
STREAM_URL = "http://172.20.10.2:81/stream"

# Camera setup
camera = None
camera_running = False
last_frame_time = 0
frame_interval = 1/60  # mục tiêu 60 FPS

# Global variables cho xử lý bất đồng bộ
global_result = None  # Kết quả YOLO (và sau đó có thể gồm thêm thông tin của R50)
processing = False    # Cờ báo trạng thái inference đang chạy hay không
frame_count = 0       # Đếm số frame

# Tracking recognition - new
recognition_history = []    # Track consecutive recognitions
recognized_employee = None  # Currently recognized employee
recognition_count = 0       # Count of consecutive recognitions

def init_camera() -> bool:
    global camera, camera_running, recognition_count, recognized_employee
    try:
        # Reset recognition state when initializing camera
        recognition_count = 0
        recognized_employee = None
        
        if camera is None:
            print("Initializing camera stream...")
            camera = cv2.VideoCapture(STREAM_URL)
            # Thêm delay để đảm bảo stream khởi động
            time.sleep(1)
            
            if not camera.isOpened():
                raise Exception("Không thể kết nối đến stream")
                
            # Thiết lập các thuộc tính camera
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            camera.set(cv2.CAP_PROP_FPS, 30)
            
            camera_running = True
            print("Camera stream initialized successfully")
            return True
    except Exception as e:
        print(f"Lỗi khi khởi tạo camera stream: {str(e)}")
        camera_running = False
        return False
    return True

def release_camera() -> None:
    global camera, camera_running, recognition_count, recognized_employee
    try:
        print("Releasing camera...")
        # Reset recognition state when releasing camera
        recognition_count = 0
        recognized_employee = None
        
        if camera is not None:
            # Release the camera
            camera.release()
            camera = None
            print("Camera released successfully")
        
        camera_running = False
        
        # Clean up servo controller
        servo_controller.cleanup()
    except Exception as e:
        print(f"Lỗi khi giải phóng camera: {str(e)}")
        camera_running = False
        # Ensure camera is set to None even if there was an error
        camera = None

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
        # Ensure all IDs are strings for consistent lookup
        employee_dict = {str(emp["id"]): emp["full_name"] for emp in employees}
        print(f"Updated employee mapping with {len(employee_dict)} employees:")
        for id_str, name in employee_dict.items():
            print(f"  ID: {id_str} -> {name}")
    except Exception as e:
        print(f"Error updating employee mapping: {str(e)}")
        import traceback
        traceback.print_exc()

# Reset recognition tracking
def reset_recognition():
    """Reset all recognition tracking variables"""
    global recognized_employee, recognition_count
    recognized_employee = None
    recognition_count = 0
    print("Recognition tracking reset")

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
    global global_result, processing, recognition_history, recognized_employee, recognition_count
    try:
        start_time = time.time()  # Ghi lại thời gian bắt đầu
        
        # Chạy inference YOLO trên yolo_frame
        res = model(yolo_frame, verbose=True)[0]
        
        # Tạo danh sách để lưu kết quả tùy chỉnh
        custom_results = []
        
        # Tính scaling factors từ khung hình YOLO đến original_frame
        scale_x = original_frame.shape[1] / yolo_frame.shape[1]
        scale_y = original_frame.shape[0] / yolo_frame.shape[0]
        
        # Variables to track the best recognized face
        current_recognized_id = None
        current_employee_data = None
        highest_prob = 0.0
        
        # Check if any faces were detected
        if len(res.boxes) == 0:
            print("No faces detected in frame")
            # If no faces detected for more than 5 frames, reset recognition counter
            # But don't reset immediately to handle occasional missed frames
            if recognition_count > 0:
                recognition_count -= 0.5  # Decrease counter gradually
                if recognition_count < 0:
                    recognition_count = 0
                print(f"No face detected, reducing count to: {recognition_count}")
            
            # Lưu kết quả trống vào global_result
            global_result = []
            processing = False
            return
        
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
                        print(f"Predicted id_user: {id_user}, type: {type(id_user)}")
                        
                        # Ensure id_user is an integer
                        try:
                            id_user_int = int(id_user)
                            id_user = id_user_int
                        except (ValueError, TypeError):
                            print(f"Warning: Unable to convert id_user {id_user} to int")
                        
                        name = employee_dict.get(str(id_user), "Unknown")
                        
                        # Track the face with highest probability for recognition history
                        if max_prob > highest_prob and id_user != "Unknown":
                            highest_prob = max_prob
                            current_recognized_id = id_user
                            current_employee_data = {"id": id_user, "name": name}
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
        
        # Update recognition history
        if current_recognized_id:
            # If the same employee is recognized as before
            if recognized_employee and recognized_employee["id"] == current_recognized_id:
                recognition_count += 1
                print(f"Consecutive recognition: {recognition_count} for ID {current_recognized_id}")
                
                # If we have 3 consecutive recognitions, set the recognized employee and open door
                if recognition_count >= 3:
                    print(f"Employee {current_recognized_id} verified after 3 consecutive recognitions")
                    # Open the door when employee is verified
                    servo_controller.open_door()
                    # Close the door after 5 seconds
                    threading.Timer(5.0, servo_controller.close_door).start()
            else:
                # Reset for new employee
                recognized_employee = current_employee_data
                recognition_count = 1
                print(f"New recognition: {current_recognized_id}")
        else:
            # No recognized face in this frame (Unknown)
            # Keep the current recognition state to avoid immediate reset
            # (recognition_count will be decreased gradually in UI only)
            pass
        
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
                                id_user = result['id_user']
                                
                                # Chuyển tọa độ sang khung hình gốc
                                x1_scaled = int(x1 * scale_x)
                                y1_scaled = int(y1 * scale_y)
                                x2_scaled = int(x2 * scale_x)
                                y2_scaled = int(y2 * scale_y)
                                
                                # Thiết lập màu dựa trên trạng thái nhận diện
                                color = (0, 255, 0)  # Xanh lá mặc định
                                if id_user == "Unknown":
                                    color = (0, 0, 255)  # Đỏ cho Unknown
                                elif recognized_employee and recognized_employee["id"] == id_user:
                                    if recognition_count >= 3:
                                        color = (0, 255, 255)  # Vàng cho đã xác minh
                                    else:
                                        color = (0, 165, 255)  # Cam cho đang xác minh
                                
                                # Vẽ bounding box
                                cv2.rectangle(frame, (x1_scaled, y1_scaled), 
                                            (x2_scaled, y2_scaled), color, 2)
                                
                                # Vẽ conf phía trên bounding box
                                conf_label = f"{conf:.2f}"
                                cv2.putText(frame, conf_label, (x1_scaled, y1_scaled - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                # Vẽ Name phía dưới bounding box
                                name_label = f"Name: {name}"
                                cv2.putText(frame, name_label, (x1_scaled, y2_scaled + 20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                           
                        # Hiển thị trạng thái nhận diện lên frame
                        if recognized_employee and recognition_count > 0:
                            # In debug thông tin nhận diện
                            print(f"Recognition info: employee_id={recognized_employee['id']}, count={recognition_count}")
                            
                            status_text = f"Đang xác minh: {recognized_employee['name']} ({int(recognition_count)}/3)"
                            if recognition_count >= 3:
                                status_text = f"Đã xác minh: {recognized_employee['name']}"
                            cv2.putText(frame, status_text, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
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
                                    
                                    # Extract and save face only if capturing is enabled
                                    face_img = frame[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
                                    if face_img.size > 0 and face_scanner.capturing:
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
                            # Draw capturing status
                            status_text = "Ready to capture" if not face_scanner.capturing else "Capturing..."
                            cv2.putText(frame, status_text, (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                                      (0, 255, 255) if not face_scanner.capturing else (0, 255, 0), 2)
                                      
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