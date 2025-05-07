import os
import cv2
import numpy as np
import onnxruntime as ort


print("DM BAT DAU TRAIN DAY ROIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
print("Bat dau chay training.py")
# Hàm tiền xử lý ảnh
def preprocess_image(image_path, input_size=(112, 112)):
    # Bước 2: Tiền xử lý hình ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image at {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
    
    # Bước 3: Resize về 112x112
    img = cv2.resize(img, input_size)
    
    # Chuẩn hóa theo ArcFace
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # Đổi sang (C, H, W)
    img = np.expand_dims(img, axis=0)   # Thêm batch dimension
    return img

# Hàm trích xuất đặc trưng
def extract_features(image_path, session):
    input_img = preprocess_image(image_path)
    if input_img is None:
        return None
    # Bước 4: Đưa vào mô hình để trích xuất đặc trưng
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    features = session.run([output_name], {input_name: input_img})[0]
    return features.flatten()

# Tải mô hình ONNX
onnx_model_path = "R50.onnx"
session = ort.InferenceSession(onnx_model_path)

# Bước 1: Lấy dữ liệu
data_dir = "face_data"
features = []
labels = []

# Duyệt qua các folder con
for user_id in os.listdir(data_dir):
    user_folder = os.path.join(data_dir, user_id)
    if os.path.isdir(user_folder):
        for img_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_name)
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"Processing {img_path}...")
                feature = extract_features(img_path, session)
                if feature is not None:
                    features.append(feature)
                    labels.append(user_id)

# Chuyển sang numpy array
features = np.array(features)
labels = np.array(labels)

# In kết quả
print(f"Extracted {len(features)} feature vectors with shape {features.shape}")


from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib

# Mã hóa nhãn (id_user) thành số
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Khởi tạo mô hình SVM
svm_model = SVC(probability=True)

# Định nghĩa lưới tham số để tìm kiếm
param_grid = {
    'kernel': ['linear', 'rbf'],  # Thử hai kernel
    'C': [0.1, 1, 10, 100],      # Thử các giá trị C
    'gamma': ['scale', 0.001, 0.01, 0.1],  # Thử gamma (chỉ ảnh hưởng với 'rbf')
    'class_weight': [None, 'balanced']  # Thử cân bằng lớp
}

# Tạo GridSearchCV
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    cv=5,  # Số fold trong cross-validation
    scoring='accuracy',  # Đánh giá bằng accuracy
    n_jobs=-1  # Sử dụng tất cả CPU để tăng tốc
)

# Huấn luyện GridSearchCV
grid_search.fit(X_train, y_train)

# In ra tham số tốt nhất
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Đánh giá trên tập kiểm tra
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test set score:", test_score)

# Lưu mô hình tốt nhất và LabelEncoder
joblib.dump(best_model, "svm_face_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print(">>> Training completed")
print("Model training completed and saved!")