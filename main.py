import os
import time

import pandas as pd
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from src.utils import dataset_factory
from src.trainer import Trainer
from src.transform import ImageTransform
from src.model import ResNet, ViTTransformer


_ROOT_DIR = "./"
_CSV_FILE = "./data/labels.csv"
_LOG_LOCATION = "./logs/log.csv"

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 6.255533209074749e-05
SCHEDULER_ALPHA = 0.00036720508404985644
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING = True
PATIENCE = 5
BEST_METRIC = "val_loss"
MIN_OR_MAX = "min"
VERBOSE = True

TRANSFORM = True
OUTPUT_SIZE = (255, 255)

def main():
    if TRANSFORM:
        transform = ImageTransform(OUTPUT_SIZE)
    else:
        transform = None

    train_loader, test_loader = dataset_factory(
        transform=transform,
        root_dir=_ROOT_DIR,
        csv_file=_CSV_FILE,
        batch_size=BATCH_SIZE
    )
    
    n_classes = pd.read_csv(_CSV_FILE)["Label"].nunique()

    # model = ResNet(
    #     n_classes=n_classes,
    #     image_input_shape=OUTPUT_SIZE,
    #     input_channels=3,
    #     resnet_blocks=2,
    #     resnet_channels=[64, 38],
    #     resnet_kernel_sizes=[5, 3],
    #     resnet_strides=[1, 1],
    #     resnet_padding_sizes=[0, 0],
    #     resnet_layers=[1, 2],
    #     fc1_output_dims=128,
    #     fc2_output_dims=128,
    #     pool_kernel_size=2,
    #     pool_stride=2,
    #     dropout=0.,
    # )
    model = ViTTransformer(
        n_classes=n_classes,
        input_channels=3,
        image_input_shape=OUTPUT_SIZE,
        n_patches=17,
        hidden_d=512,
        n_heads=32,
        dropout=0.1,
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = LinearLR(optimizer, SCHEDULER_ALPHA)

    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_location = f"./models/{model.__class__.__name__}/{now}"
    if not os.path.exists("./models"):
        os.mkdir("./models")
    if not os.path.exists(f"./models/{model.__class__.__name__}"):
        os.mkdir(f"./models/{model.__class__.__name__}")

    additional_reporting = {
        "model": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "loss_fn": loss_fn.__class__.__name__,
        "scheduler": scheduler.__class__.__name__ if scheduler else None,
        "scheduler_alpha": SCHEDULER_ALPHA,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "early_stopping": EARLY_STOPPING,
        "patience": PATIENCE,
        "best_metric": BEST_METRIC,
        "min_or_max": MIN_OR_MAX,
        "transform": TRANSFORM,
        "transform_output_size": OUTPUT_SIZE,
        "model_save_path": save_location,
    }

    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=EPOCHS,
        save_location=save_location,
        training_log=_LOG_LOCATION,
        scheduler=scheduler,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        early_stopping_metric=BEST_METRIC,
        min_or_max=MIN_OR_MAX,
        verbose=VERBOSE,
        additional_reporting=additional_reporting,
    )

    trainer.train()

if __name__=="__main__":
    main()