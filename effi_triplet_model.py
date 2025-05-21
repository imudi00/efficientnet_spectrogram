import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import os
import random

# 1) EfficientNet 임베딩 모델 정의
class EfficientNetEmbedding(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        
        # 1. EfficientNet-B0 불러오기 (pretrained)
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        # 2. EfficientNet 파라미터 동결
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 3. 필요한 부분만 추출
        self.features = self.base_model.features  # feature extractor
        self.pool = nn.AdaptiveAvgPool2d(1)       # 글로벌 평균 풀링
        self.embedding = nn.Linear(1280, embedding_size)  # 임베딩 레이어
        self.l2_norm = nn.functional.normalize     # L2 정규화
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = self.l2_norm(x, dim=1)
        return x

# 2) Triplet loss 및 optimizer 설정
model = EfficientNetEmbedding(embedding_size=128)
loss_fn = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3) Triplet Dataset 클래스 정의
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.song_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        self.data = []
        for song_dir in self.song_dirs:
            images = [f for f in os.listdir(song_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            if len(images) >= 2:
                self.data.append((song_dir, images))
        
    def __len__(self):
        return sum(len(images) for _, images in self.data)
    
    def __getitem__(self, idx):
        anchor_song_idx = random.randint(0, len(self.data) - 1)
        anchor_song_dir, anchor_images = self.data[anchor_song_idx]

        anchor_img_name = random.choice(anchor_images)
        positive_img_name = anchor_img_name
        while positive_img_name == anchor_img_name:
            positive_img_name = random.choice(anchor_images)

        negative_song_idx = anchor_song_idx
        while negative_song_idx == anchor_song_idx:
            negative_song_idx = random.randint(0, len(self.data) - 1)
        negative_song_dir, negative_images = self.data[negative_song_idx]
        negative_img_name = random.choice(negative_images)

        anchor_img = Image.open(os.path.join(anchor_song_dir, anchor_img_name)).convert('RGB')
        positive_img = Image.open(os.path.join(anchor_song_dir, positive_img_name)).convert('RGB')
        negative_img = Image.open(os.path.join(negative_song_dir, negative_img_name)).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# 4) 이미지 전처리 설정
transform = transforms.Compose([
    #전에 하던 증강이랑 똑같이 처리.
        transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# 5) 데이터셋과 데이터로더 생성
root_data_dir = "./data"  # 자신의 데이터 경로로 변경하세요 ✅✅✅✅ 입력 필요
triplet_dataset = TripletDataset(root_dir=root_data_dir, transform=transform)
data_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True, num_workers=4)

# 6) Triplet 학습 루프 함수
def train_triplet(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for anchor, positive, negative in data_loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)
        
        loss = loss_fn(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)

# 7) 임베딩 추출 함수 (갤러리 및 테스트 모두 사용 가능)
def extract_embeddings(model, inputs, device, batch_size=64):
    """
    inputs: DataLoader 또는 Tensor(single or batch)
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        if isinstance(inputs, DataLoader):
            for batch in inputs:
                batch = batch.to(device)
                emb = model(batch)
                embeddings.append(emb.cpu().numpy())
            embeddings = np.vstack(embeddings)
        else:
            # 단일 또는 소량 이미지 텐서 처리
            inputs = inputs.to(device)
            emb = model(inputs)
            embeddings = emb.cpu().numpy()

    return embeddings

# 8) Top-K 추천 함수
def recommend_topk(query_embedding, gallery_embeddings, gallery_ids, topk=5):
    sims = cosine_similarity(query_embedding.reshape(1, -1), gallery_embeddings).flatten()
    topk_idx = sims.argsort()[::-1][:topk]
    return [(gallery_ids[i], sims[i]) for i in topk_idx]

# 9) 학습 및 추천 실행
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 10
    for epoch in range(num_epochs):
        avg_loss = train_triplet(model, data_loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # 체크포인트 저장 (필요시)
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    # 갤러리 임베딩 미리 추출 및 저장
    gallery_loader = DataLoader(triplet_dataset, batch_size=64, shuffle=False, num_workers=4)
    gallery_embeddings = extract_embeddings(model, gallery_loader, device)
    np.save("gallery_embeddings.npy", gallery_embeddings)

    # 테스트 쿼리 임베딩 및 추천
    test_img_path = "./test_query.png"  # 사용자 쿼리 이미지 경로 ✅✅✅✅✅ 입력 필요
    test_img = Image.open(test_img_path).convert('RGB')
    test_img_tensor = transform(test_img).unsqueeze(0)  # 배치 차원 추가

    query_embedding = extract_embeddings(model, test_img_tensor, device)

    gallery_ids = [d for d, _ in triplet_dataset.data]  # 곡 ID 리스트 (폴더명)
    recommendations = recommend_topk(query_embedding, gallery_embeddings, gallery_ids, topk=5)
    print("추천 결과:", recommendations)
