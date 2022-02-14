import os
import glob
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FaceDataset(Dataset):
    def __init__(self, dataset_root_list, same_prob=0.2):
        datasets = []
        self.N = []
        self.same_prob = same_prob

        imgpaths_in_root = []
        for dataset_root in dataset_root_list:
            for root, dirs, files in os.walk(dataset_root):
                for dir in dirs:
                    imgpaths_in_root += glob.glob(f'{root}/{dir}/*.*g')

            datasets.append(imgpaths_in_root)

        self.N.append(len(imgpaths_in_root))
        self.datasets = datasets
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
        image_path = self.datasets[idx][item]
        
        Xs = Image.open(image_path).convert("RGB")

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = Image.open(image_path).convert("RGB")
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.N)

