import torch
from torch.utils import data
from PIL import Image
import os
import random

def pil_loader(path: str) -> Image.Image:
    '''
    From https://github.com/pytorch/vision/blob/65676b4ba1a9fd4417293cb16f690d06a4b2fb4b/torchvision/datasets/folder.py#L244
    '''
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class DripData(data.Dataset):
    def __init__(self, image_dir, ann_path, img_transform=None, train=True, train_pct=1.0):
        self.image_dir = image_dir
        self.ann_path = ann_path

        files = [file for file in os.listdir(image_dir) if file.endswith('png')]
        if train:
            files = files[:int(train_pct * len(files))]
        else:
            files = files[int(train_pct * len(files)):]
        self.images = {}
        for file in files:
            file_path = os.path.join(image_dir, file)
            img = pil_loader(file_path)
            if img_transform is not None:
                img = img_transform(img)
            img_id = int(file[:-4])
            self.images[img_id] = img

        self.anns = {}
        with open(ann_path) as f:
            anns = f.read()
        anns = anns.strip().splitlines()
        self.cls2idx = dict()
        for ann in anns:
            img_id, cls = ann.strip().split(',')
            img_id = int(img_id)
            if img_id not in self.images:
                continue
            idx = self.cls2idx.get(cls)
            if idx is None:
                idx = len(self.cls2idx)
                self.cls2idx[cls] = idx
            self.anns[img_id] = idx
        count = self.rand_fill_anns()
        if count > 0:
            print(f'Randomly filled in {count} annotations :)')

        self.idx2img = list(self.images.keys())
    
    def rand_fill_anns(self):
        count = 0
        for img_id in self.images:
            if img_id not in self.anns:
                self.anns[img_id] = random.randint(1, 4)
                count += 1
        return count

    def __getitem__(self, i):
        img_id = self.idx2img[i]
        return self.images[img_id], self.anns[img_id]

    def __len__(self):
        return len(self.idx2img)

    def collate_fn(batch):
        xs, ys = [], []
        for x, y in batch:
            xs.append(x)
            ys.append(y)
        return torch.stack(xs, dim=0), torch.tensor(ys)

def make_loaders(args, image_dir, ann_path, img_transform=None, train_pct=0.8):
    train_dataset = DripData(image_dir, ann_path, img_transform=img_transform, 
        train=True, train_pct=train_pct)
    test_dataset = DripData(image_dir, ann_path, img_transform=img_transform, 
        train=False, train_pct=train_pct)

    train_loader_args = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'collate_fn': DripData.collate_fn,
        'num_workers': 4 if args.gpu else 0
    }
    test_loader_args = train_loader_args.copy()
    test_loader_args['shuffle'] = False

    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    test_loader = data.DataLoader(test_dataset, **test_loader_args)
    return train_loader, test_loader

if __name__ == '__main__':
    from torchvision import transforms
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    data = DripData('../data_collection/data/images', 
                    '../data_collection/annotations.txt', 
                    img_transform=img_transform)
