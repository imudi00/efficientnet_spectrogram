import os
import random
from torch.utils.data import Dataset
from PIL import Image

class TripletDataset(Dataset):
    def __init__(self, root_dir, song_ids=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if song_ids is None:
            song_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.song_data = {}
        for song_id in song_ids:
            song_path = os.path.join(root_dir, song_id)
            image_files = [f for f in os.listdir(song_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(image_files) >= 2:
                self.song_data[song_id] = [os.path.join(song_path, f) for f in image_files]
        self.song_ids = list(self.song_data.keys())

    def __len__(self):
        return sum(len(v) for v in self.song_data.values())

    def __getitem__(self, idx):
        anchor_song_id = random.choice(self.song_ids)
        candidates = self.song_data[anchor_song_id]
        anchor_path = random.choice(candidates)
        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = random.choice(candidates)
        negative_song_id = random.choice([s for s in self.song_ids if s != anchor_song_id])
        negative_path = random.choice(self.song_data[negative_song_id])

        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img