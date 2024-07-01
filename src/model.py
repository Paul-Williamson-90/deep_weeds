import json

import torch
from torch import nn
from torch.nn import functional as F


class SimpleCNN(nn.Module):
    def __init__(
            self,
            n_classes: int = 3,
            image_input_shape: tuple|list = (256, 256),
            conv1_in_channels: int = 3,
            conv1_out_channels: int = 6,
            conv1_kernel_size: int = 5,
            conv1_stride: int = 1,
            conv1_padding_size: int = 0,
            conv2_out_channels: int = 16,
            conv2_kernel_size: int = 5,
            conv2_padding_size: int = 0,
            conv2_stride: int = 1,
            pool_kernel_size: int = 2,
            pool_stride: int = 2,
            fc1_output_dims: int = 120,
            fc2_output_dims: int = 84,
        ):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=conv1_in_channels, 
            out_channels=conv1_out_channels, 
            kernel_size=conv1_kernel_size,
            padding=conv1_padding_size,
            stride=conv1_stride
        )
        output_size = self._calculate_output_size_conv(
            input_size=image_input_shape[0],
            kernel_size=conv1_kernel_size,
            padding_size=conv1_padding_size,
            stride=conv1_stride
        )
        
        self.pool = nn.MaxPool2d(
            kernel_size=pool_kernel_size, 
            stride=pool_stride
        )
        output_size = self._calculate_output_size_pool(
            input_size=output_size,
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels, 
            out_channels=conv2_out_channels, 
            kernel_size=conv2_kernel_size,
            padding=conv2_padding_size,
            stride=conv2_stride
        )
        output_size = self._calculate_output_size_conv(
            input_size=output_size,
            kernel_size=conv2_kernel_size,
            padding_size=conv2_padding_size,
            stride=conv2_stride
        ) 
        output_size = self._calculate_output_size_pool(
            input_size=output_size,
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
    
        self.fc1_input_dims = conv2_out_channels * output_size * output_size

        self.fc1 = nn.Linear(self.fc1_input_dims, fc1_output_dims)
        self.fc2 = nn.Linear(fc1_output_dims, fc2_output_dims)
        self.fc3 = nn.Linear(fc2_output_dims, n_classes)
        self.output_activation = nn.Softmax(dim=1)
        self.config = {
            "n_classes": n_classes,
            "image_input_shape": image_input_shape,
            "conv1_in_channels": conv1_in_channels,
            "conv1_out_channels": conv1_out_channels,
            "conv1_kernel_size": conv1_kernel_size,
            "conv1_padding_size": conv1_padding_size,
            "conv1_stride": conv1_stride,
            "conv2_out_channels": conv2_out_channels,
            "conv2_kernel_size": conv2_kernel_size,
            "conv2_padding_size": conv2_padding_size,
            "conv2_stride": conv2_stride,
            "pool_kernel_size": pool_kernel_size,
            "pool_stride": pool_stride,
            "fc1_output_dims": fc1_output_dims,
            "fc2_output_dims": fc2_output_dims,
        }

    def _calculate_output_size_conv(self, input_size, kernel_size, padding_size, stride):
        return ((input_size - kernel_size + 2 * padding_size) // stride) + 1
    
    def _calculate_output_size_pool(self, input_size, kernel_size, stride):
        return ((input_size - kernel_size) // stride) + 1

    def forward(self, x):
        x = x / 255.0 # Normalize
        x = F.relu(self.conv1(x)) 
        x = self.pool(x) 
        x = F.relu(self.conv2(x)) 
        x = self.pool(x) 
        x = x.view(x.size(0), -1)
        x = self.fc1(x) 
        x = F.relu(x) 
        x = self.fc2(x) 
        x = F.relu(x) 
        x = self.fc3(x) 
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.output_activation(self.forward(x))
        return x
    
    def save_pretrained(self, save_directory: str):
        with open(f"{save_directory}/config.json", "w") as f:
            json.dump(self.config, f, indent=4)
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, save_directory: str):
        with open(f"{save_directory}/config.json", "r") as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(f"{save_directory}/pytorch_model.bin"))
        return model