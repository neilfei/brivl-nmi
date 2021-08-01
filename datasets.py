import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import AutoTokenizer


class ImageData(Dataset):
    def __init__(self, cfg, data_root):
        self.cfg = cfg
        csv_path = './input_data/images.csv'
        lines = [x.strip() for x in open(csv_path, 'r').readlines()]

        data = []
        for l in lines:
            path = os.path.join(data_root, l)
            data.append(path)

        self.data = data

        image_size = cfg.MODEL.IMG_SIZE
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            normalize
        ])

    def __getitem__(self, i):
        path = self.data[i]
        image = self.transform(Image.open(path).convert('RGB'))

        img_box_s = []
        new_size = self.cfg.MODEL.IMG_SIZE
        box_grid = self.cfg.MODEL.BOX_GRID
        for i in range(box_grid):
            for j in range(box_grid):
                img_box_s.append(torch.from_numpy(np.array([i * (new_size / box_grid), j * (new_size / box_grid), (i+1) * (new_size / box_grid), (j+1) * (new_size / box_grid)])))
        img_box_s.append(torch.from_numpy(np.array([0, 0, new_size, new_size]).astype(np.float32))) # bbox number:  self.cfg.MODEL.MAX_IMG_LEN

        valid_len = len(img_box_s)
        img_len = torch.full((1,), valid_len, dtype=torch.long)

        if valid_len < self.cfg.MODEL.MAX_IMG_LEN:
            for i in range(self.cfg.MODEL.MAX_IMG_LEN - valid_len):
                img_box_s.append(torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)))

        image_boxs = torch.stack(img_box_s, 0) # <36, box_grid>
        
        return image, img_len, image_boxs

    def __len__(self):
        return len(self.data)


class TextData(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        csv_path = './input_data/texts.csv'
        self.data = [x.strip() for x in open(csv_path, 'r').readlines()]

        self.text_transform = AutoTokenizer.from_pretrained(cfg.MODEL.ENCODER)
        self.max_text_len = cfg.MODEL.MAX_TEXT_LEN

    def __getitem__(self, i):
        text = self.data[i]
        text_info = self.text_transform(text, padding='max_length', truncation=True,
                                        max_length=self.max_text_len, return_tensors='pt')
        text = text_info.input_ids.reshape(-1)
        text_len = torch.sum(text_info.attention_mask)
        
        return text, text_len

    def __len__(self):
        return len(self.data)









