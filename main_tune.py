import os
import time
import logging

import pandas as pd
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

from src.utils import dataset_factory
from src.trainer import Trainer
from src.transform import ImageTransform
from src.model import SimpleCNN, ResNet, ViTTransformer

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

CWD = os.getcwd()

# Define constants
EPOCHS = 50
PATIENCE = 5
BEST_METRIC = "f1"
MIN_OR_MAX = "max"
OUTPUT_SIZE = (255, 255)
LOG_LOCATION = f"{CWD}/logs/log.csv"

def load_data(transform, root_dir, csv_file, batch_size):
    train_loader, test_loader = dataset_factory(
        transform=transform,
        root_dir=root_dir,
        csv_file=csv_file,
        batch_size=batch_size
    )
    n_classes = pd.read_csv(csv_file)["Label"].nunique()
    return train_loader, test_loader, n_classes

def objective(config):
    # try:
        transform = ImageTransform(OUTPUT_SIZE)
        root_dir = config["root_dir"]
        csv_file = config["csv_file"]
        batch_size = config["batch_size"]

        train_loader, test_loader, n_classes = load_data(transform, root_dir, csv_file, batch_size)

        # model = SimpleCNN(
        #     n_classes=n_classes,
        #     image_input_shape=OUTPUT_SIZE,
        #     conv1_in_channels=3,
        #     conv1_out_channels=config["conv1_out_channels"],
        #     conv1_kernel_size=config["conv1_kernel_size"],
        #     conv1_stride=config["conv1_stride"],
        #     conv1_padding_size=config["conv1_padding_size"],
        #     conv2_out_channels=config["conv2_out_channels"],
        #     conv2_kernel_size=config["conv2_kernel_size"],
        #     conv2_padding_size=config["conv2_padding_size"],
        #     conv2_stride=config["conv2_stride"],
        #     pool_kernel_size=config["pool_kernel_size"],
        #     pool_stride=config["pool_stride"],
        #     fc1_output_dims=config["fc1_output_dims"],
        #     fc2_output_dims=config["fc2_output_dims"]
        # )
        # model = ResNet(
        #     n_classes=n_classes,
        #     image_input_shape=OUTPUT_SIZE,
        #     input_channels=3,
        #     resnet_blocks=config["resnet_blocks"],
        #     resnet_channels=[config["resnet_channels"][i] for i in range(config["resnet_blocks"])],
        #     resnet_kernel_sizes=[config["resnet_kernel_sizes"][i] for i in range(config["resnet_blocks"])],
        #     resnet_strides=[1 for _ in range(config["resnet_blocks"])],
        #     resnet_padding_sizes=[0 for _ in range(config["resnet_blocks"])],
        #     resnet_layers=[config["resnet_layers"][i] for i in range(config["resnet_blocks"])],
        #     fc1_output_dims=config["fc1_output_dims"],
        #     fc2_output_dims=config["fc2_output_dims"],
        #     pool_kernel_size=2,
        #     pool_stride=2
        # )

        hidden_d, n_heads = config["hidden_d_num_heads"]
        model = ViTTransformer(
            n_classes=n_classes,
            input_channels=3,
            image_input_shape=OUTPUT_SIZE,
            hidden_d=hidden_d,
            n_patches=config["n_patches"],
            n_heads=n_heads,
            dropout=config["dropout"],
        )

        optimizer = Adam(model.parameters(), lr=config["lr"])
        loss_fn = nn.CrossEntropyLoss()
        scheduler = LinearLR(optimizer, config["scheduling_alpha"])

        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        save_location = f"{CWD}/models/{model.__class__.__name__}/{now}"
        n = 1
        while os.path.exists(save_location):
            save_location += f"_{n}"
            n += 1
        os.makedirs(save_location, exist_ok=True)

        additional_reporting = {
            "model": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "loss_fn": loss_fn.__class__.__name__,
            "scheduler": scheduler.__class__.__name__ if scheduler else None,
            "scheduler_alpha": config["scheduling_alpha"],
            "batch_size": batch_size,
            "learning_rate": config["lr"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "early_stopping": True,
            "patience": 5,
            "best_metric": "val_loss",
            "min_or_max": "min",
            "transform": True,
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
            training_log=LOG_LOCATION,
            scheduler=scheduler,
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            early_stopping=True,
            patience=PATIENCE,
            early_stopping_metric="val_loss",
            min_or_max="min",
            verbose=False,
            additional_reporting=additional_reporting,
        )

        stop_training = False
        while not stop_training:
            for epoch in range(trainer.n_epochs):
                stop_training, metrics = trainer._epoch(epoch + 1)
                train.report(metrics)
                if stop_training:
                     break
            break
        trainer._save_best_metrics()
        train.report(trainer.best_metrics)
    # except Exception:
    #     train.report({
    #         "train_loss": float("inf"),
    #         "val_loss": float("inf"),
    #         "n_epoch": 0,
    #         "f1": 0.0,
    #         "acc": 0.0,
    #         "precision": 0.0,
    #         "recall": 0.0,
    #     })
    #     if os.path.exists(save_location):
    #         os.rmdir(save_location)

def main():
    search_space = { # ViTTransformer
        "hidden_d_num_heads": tune.choice(
            [[64, 8], [128, 8], [256, 8], [512, 8], [64, 16], [128, 16], [256, 16], [512, 16], [64, 32], [128, 32], [256, 32], [512, 32]]
        ),
        "n_patches": tune.choice([5, 15, 17, 51, 85]),
        "dropout": tune.uniform(0., 0.5),
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([8, 16, 32]),
        "gradient_accumulation_steps": tune.choice([1, 2, 4, 8]),
        "scheduling_alpha": tune.loguniform(1e-6, 1e-1),
        "root_dir": tune.choice([f"{CWD}/"]),  
        "csv_file": tune.choice([f"{CWD}/data/labels.csv"]), 
    }
    # search_space = { # ResNet
    #     "resnet_blocks": tune.choice([1, 2]),
    #     "resnet_channels": tune.choice([[6, 12], [12, 24], [24, 48]]),
    #     "resnet_kernel_sizes": tune.choice(
    #         [
    #               [5, 1],[5, 3],[5, 2], [4, 3], [4, 2], [4, 1], [3, 2], [3, 1], [2, 1], [1, 1], [2, 2], [2, 3], [3, 3]
    #         ]
    #     ),
    #     "resnet_layers": tune.choice([[1, 1],[1, 2], [2, 1], [2, 2]]),
    #     "fc1_output_dims": tune.choice([64, 128, 256]),
    #     "fc2_output_dims": tune.choice([64, 128, 256]),
    #     "lr": tune.loguniform(1e-6, 1e-1),
    #     "batch_size": tune.choice([8, 16, 32]),
    #     "gradient_accumulation_steps": tune.choice([1, 2, 4, 8]),
    #     "scheduling_alpha": tune.loguniform(1e-6, 1e-1),
    #     "root_dir": tune.choice([f"{CWD}/"]),  
    #     "csv_file": tune.choice([f"{CWD}/data/labels.csv"]), 
    # }
    # search_space = { # SimpleCNN
    #     "conv1_out_channels": tune.choice([12, 24, 48]),
    #     "conv1_kernel_size": tune.choice([5, 10, 15]),
    #     "conv1_padding_size": tune.choice([0, 5, 10]),
    #     "conv1_stride": tune.choice([1, 2, 5]),
    #     "conv2_out_channels": tune.choice([6, 12, 24]),
    #     "conv2_kernel_size": tune.choice([5, 10, 15]),
    #     "conv2_padding_size": tune.choice([0, 5, 10]),
    #     "conv2_stride": tune.choice([1, 2, 5]),
    #     "pool_kernel_size": tune.choice([5, 10, 15]),
    #     "pool_stride": tune.choice([1, 2, 5]),
    #     "fc1_output_dims": tune.choice([64, 128, 256]),
    #     "fc2_output_dims": tune.choice([64, 128, 256]),
    #     "lr": tune.loguniform(1e-6, 1e-1),
    #     "batch_size": tune.choice([8, 16, 32]),
    #     "gradient_accumulation_steps": tune.choice([1, 2, 4, 8]),
    #     "scheduling_alpha": tune.loguniform(1e-6, 1e-1),
    #     "root_dir": tune.choice([f"{CWD}/"]),  
    #     "csv_file": tune.choice([f"{CWD}/data/labels.csv"]), 
    # }

    algo = OptunaSearch()

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric=BEST_METRIC,
            mode=MIN_OR_MAX,
            search_alg=algo,
            num_samples=50,
            max_concurrent_trials=3,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    logging.info("Best config is: %s", results.get_best_result().config)

if __name__ == "__main__":
    main()
