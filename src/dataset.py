import os
import torch
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


class WeedsDataset(Dataset):

    def __init__(
            self, 
            csv_file:pd.DataFrame, 
            root_dir:str, 
            transform=None,
        ):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.root_dir,
            self.csv_file.loc[idx, "file_path"]
        )

        image = io.imread(img_name)
        label = self.csv_file.loc[idx, "Label"]
        sample = {
            "image": image,
            "label": label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
