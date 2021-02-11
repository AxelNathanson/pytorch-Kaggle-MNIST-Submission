import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from torchvision import transforms
from image_dataset import ImageDataset

torch.set_default_dtype(torch.float)


def load_dataset(csv_file, **kwargs):
    """
    Function to load in the Kaggle-version of the MNIST-dataset.
    
    Arguments:
        csv_file {string} -- location of csv-file

    Optional Arguments:
        split {float} -- Training-set split
    Returns:
        [tuple] -- (training_loader, validation_loader)
    """

    split = kwargs.pop('split', 0.85)

    normalize = transforms.Normalize(mean=(0.1307,), 
                                     std=(0.3081,),
                                     inplace=True)

    transform_train = transforms.Compose([transforms.RandomRotation(15, fill=(0,)), 
                                          transforms.ToTensor(),
                                          normalize])

    transform_val = transforms.Compose([transforms.ToTensor(),
                                        normalize])

    data = pd.read_csv(csv_file).to_numpy()

    if data.shape[1] == 785:
        target = data[:, 0]
        images = np.reshape(data[:, 1:], (-1, 28, 28))
        train_img, val_img, train_labels, val_labels = train_test_split(images, 
                                                                        target, 
                                                                        train_size=split)
        
        train_set = ImageDataset((train_img, train_labels), transform_train)
        val_set = ImageDataset((val_img, val_labels), transform_val)

        return train_set, val_set

    elif data.shape[1] == 784:
        images = np.reshape(data, (-1, 28, 28))
        return ImageDataset(images, transform_val)
    else:
        print('Wrong Image-size for this architecture. Expected size 28x28.')
