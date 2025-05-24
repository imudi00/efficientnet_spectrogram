import os
from torch.utils.data import Dataset
from PIL import Image

class SingleImageDataset(Dataset):
    def __init__(self, root_dir, song_ids=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.image_ids = []
        song_dirs = [os.path.join(root_dir, d) for d in song_ids]
        for song_dir in song_dirs:
            images = [f for f in os.listdir(song_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in images:
                self.image_paths.append(os.path.join(song_dir, img))
                self.image_ids.append(os.path.basename(song_dir))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img