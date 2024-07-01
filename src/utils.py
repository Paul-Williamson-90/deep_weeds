import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import WeedsDataset

def dataset_factory(
        transform = None,
        root_dir:str = "./",
        csv_file:str = "./data/labels.csv",
        batch_size:int = 16,
):
    csv_file = pd.read_csv(csv_file)
    train_csv = csv_file[csv_file["split"]=="train"].copy().reset_index(drop=True)
    test_csv = csv_file[csv_file["split"]=="test"].copy().reset_index(drop=True)

    train_dataset = WeedsDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        transform=transform
    )
    test_dataset = WeedsDataset(
        csv_file=test_csv,
        root_dir=root_dir,
        transform=transform.no_augment() if transform else None
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader