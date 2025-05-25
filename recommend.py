import os
import random
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from models.embedding_model import EfficientNetEmbedding
from datasets.single_image_dataset import SingleImageDataset
from utils.inference import extract_embeddings
from configs import config
from sklearn.metrics.pairwise import cosine_similarity

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = EfficientNetEmbedding(embedding_size=config.embedding_size)
model_path = os.path.join(config.model_save_dir, "model_epoch_100.pth")  # 또는 원하는 epoch
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gallery_ids = config.train_ids + config.val_ids
gallery_dataset = SingleImageDataset(root_dir=config.data_root, song_ids=gallery_ids, transform=transform)
gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False, num_workers=0)
gallery_embeddings = extract_embeddings(model, gallery_loader, device)
gallery_id_names = gallery_dataset.image_ids

test_img_dir = config.test_root
all_test_paths = [
    os.path.join(test_img_dir, f)
    for f in os.listdir(test_img_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]
random.seed(42)
test_img_paths = random.sample(all_test_paths, min(4, len(all_test_paths)))

output_file_path = os.path.join(config.save_dir, "recommendation_results.txt")
os.makedirs(config.save_dir, exist_ok=True)

with open(output_file_path, "w", encoding="utf-8") as f:
    for test_img_path in test_img_paths:
        test_img = Image.open(test_img_path).convert('RGB')
        test_img_tensor = transform(test_img).unsqueeze(0).to(device)
        query_embedding = extract_embeddings(model, test_img_tensor, device)
        sims = cosine_similarity(query_embedding, gallery_embeddings)[0]
        topk_idx = sims.argsort()[::-1][:10]

        filename = os.path.basename(test_img_path)
        true_song_id = filename.split('_')[0]

        f.write(f"\n🎧 [{os.path.basename(test_img_path)}]에 대한 추천 결과 (개별 커버 이미지 기반):\n")
        for rank, i in enumerate(topk_idx, 1):
            song_id = gallery_dataset.image_ids[i]
            image_name = os.path.basename(gallery_dataset.image_paths[i])
            similarity = sims[i]
            f.write(f"{rank}. 곡 ID: {song_id} | 이미지: {image_name} | 유사도: {similarity:.4f}\n")

        f.write(f"정답 곡 ID: {true_song_id}\n같은 곡 ID [{true_song_id}]의 다른 커버 이미지들과의 유사도:\n")
        found = False
        for i, song_id in enumerate(gallery_dataset.image_ids):
            if song_id == true_song_id:
                image_name = os.path.basename(gallery_dataset.image_paths[i])
                similarity = sims[i]
                f.write(f"    - 이미지: {image_name} | 유사도: {similarity:.4f}\n")
                found = True
        if not found:
            f.write(f"곡 ID [{true_song_id}] 커버곡을 찾지 못했습니다.\n")

print(f"\n추천 결과 '{output_file_path}' 저장장")