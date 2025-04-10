# face_recognition_triplet.py

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Dataset sinh triplet
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_images = {}
        self.image_paths = []

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                imgs = [os.path.join(class_path, f) for f in os.listdir(class_path)]
                self.class_to_images[class_name] = imgs
                self.image_paths.extend(imgs)

    def __getitem__(self, index):
        anchor_path = self.image_paths[index]
        anchor_class = anchor_path.split(os.sep)[-2]
        positive_path = random.choice([
            img for img in self.class_to_images[anchor_class] if img != anchor_path
        ])
        negative_class = random.choice([
            c for c in self.class_to_images.keys() if c != anchor_class
        ])
        negative_path = random.choice(self.class_to_images[negative_class])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return len(self.image_paths)

# 2. Mạng backbone + embedding
class FaceNetEmbedding(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceNetEmbedding, self).__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.Identity()
        self.backbone = base_model
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x

# 3. Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# 4. Train pipeline

def train_model(dataset_path, embedding_size=128, epochs=10, batch_size=32):
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = TripletFaceDataset(dataset_path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FaceNetEmbedding(embedding_size).to(device)
    loss_fn = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = loss_fn(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "face_recognition_model.pth")
    print("\n✅ Mô hình đã được lưu: face_recognition_model.pth")

# 5. Hàm trích vector embedding

def extract_embedding(model_path, image_path, embedding_size=128):
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceNetEmbedding(embedding_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze().cpu().numpy()

# 6. Dự đoán nhân viên từ ảnh mới

def predict_identity(model_path, image_path, db_path="face_db.pkl", threshold=0.6):
    query_vector = extract_embedding(model_path, image_path)
    with open(db_path, 'rb') as f:
        database = pickle.load(f)

    min_dist = float('inf')
    identity = "Unknown"

    for person_id, vectors in database.items():
        for vec in vectors:
            dist = np.linalg.norm(query_vector - vec)
            if dist < min_dist:
                min_dist = dist
                identity = person_id

    return identity if min_dist < threshold else "Unknown"

# 7. Thêm nhân viên mới vào database

def add_to_database(model_path, image_paths, person_id, db_path="face_db.pkl"):
    embeddings = [extract_embedding(model_path, img_path) for img_path in image_paths]

    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            database = pickle.load(f)
    else:
        database = {}

    if person_id not in database:
        database[person_id] = []

    database[person_id].extend(embeddings)

    with open(db_path, 'wb') as f:
        pickle.dump(database, f)
    print(f"✅ Đã thêm {person_id} vào database")

# -----
# 7. Thêm nhân viên mới vào database (thêm cả thư mục chứa ảnh của người đó)
def add_to_database(model_path, person_folder, person_id=None, db_path="face_db.pkl"):
    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceNetEmbedding().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if person_id is None:
        person_id = os.path.basename(person_folder)

    embeddings = []
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        if not img_file.lower().endswith(('jpg', 'jpeg', 'png')):
            continue
        try:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(image)
            embeddings.append(embedding.squeeze().cpu().numpy())
        except:
            print(f"⚠️ Lỗi ảnh: {img_path}, bỏ qua...")

    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            database = pickle.load(f)
    else:
        database = {}

    if person_id not in database:
        database[person_id] = []

    database[person_id].extend(embeddings)

    with open(db_path, 'wb') as f:
        pickle.dump(database, f)
    print(f"✅ Đã thêm {person_id} ({len(embeddings)} ảnh) vào database")

# -----


# 8. Đánh giá mô hình với tập test

def evaluate_model(model_path, test_folder, db_path="face_db.pkl", threshold=0.6):
    y_true, y_pred = [], []
    for person_id in os.listdir(test_folder):
        person_path = os.path.join(test_folder, person_id)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            pred_id = predict_identity(model_path, img_path, db_path, threshold)
            y_true.append(person_id)
            y_pred.append(pred_id)

    acc = accuracy_score(y_true, y_pred)
    print("\n✅ Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(set(y_true))), labels=sorted(set(y_true)), rotation=90)
    plt.yticks(ticks=np.arange(len(set(y_true))), labels=sorted(set(y_true)))
    plt.tight_layout()
    plt.show()

# Ví dụ chạy huấn luyện
# train_model("face_train")

# Ví dụ thêm người mới vào database
# add_to_database("face_recognition_model.pth", ["face_train/001_NguyenVanA/a1.jpg"], "001_NguyenVanA")

# Ví dụ nhận diện khuôn mặt
# identity = predict_identity("face_recognition_model.pth", "face_test/001_NguyenVanA/a2.jpg")
# print("Predicted ID:", identity)

# Ví dụ đánh giá toàn bộ tập test
# evaluate_model("face_recognition_model.pth", "face_test")
