import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_data, transform = None):

        self.transform = transform
        if isinstance(image_data, tuple):
            self.train = True
            imageset = image_data[0]
            self.labels = image_data[1]
        else: 
            self.train = False
            imageset = image_data

        self.dataset = []

        for image in imageset:
            self.dataset.append(Image.fromarray(image.astype(np.uint8)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]

        if self.transform:
            image = self.transform(image)

        if self.train:
            label = self.labels[index]
            return image, label
        else:
            return image
    
        

    

    

