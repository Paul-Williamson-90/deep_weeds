import json

import torch
from torch import nn
from torch.nn import functional as F

import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return attn_probs
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        
        attn_probs = self.scaled_dot_product_attention(Q, K, mask)
        return attn_probs

class ResAttentionNetBlock(nn.Module):

    def __init__(
            self, 
            input_size:int,
            input_channels:int,
            n_channels:int, 
            kernel_size:int, 
            stride:int, 
            padding:int,
            layers:int = 2,
            pool_kernel_size:int = 2,
            pool_stride:int = 2,
            dropout: float = 0.2,
        ):
        super(ResAttentionNetBlock, self).__init__()
        self.layers = layers
        self.input_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.batch_norm_initial = nn.BatchNorm2d(n_channels)
        self.pool_initial = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        output_size = self._calculate_output_size_conv(
            input_size=input_size,
            kernel_size=kernel_size,
            padding_size=padding,
            stride=stride
        )
        output_size = self._calculate_output_size_pool(
            input_size=output_size,
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        self.conv = nn.ModuleList([nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ) for i in range(layers)])
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm2d(n_channels) 
            for _ in range(layers)
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        input_size = output_size
        output_size = self._calculate_output_size_flow(
            input_size=output_size,
            kernel_size=kernel_size,
            padding_size=padding,
            stride=stride,
            resnet_layers=layers,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            with_pool=False,
        )
        if input_size != output_size:
            self.downsample = True
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=int(((input_size - 1) - (output_size - 1) * 1 + 2 * 0) / 1 + 1),
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(n_channels)
            )
        else:
            self.downsample = False
            self.downsample_layer = None
        self.pool_out = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        self.output_size = self._calculate_output_size_pool(
            input_size=output_size,
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        self.attention = nn.ModuleList(
            [MultiHeadAttention(output_size, 1) for _ in range(n_channels)]
        )

    def _calculate_output_size_flow(
            self,
            input_size,
            kernel_size,
            stride,
            padding_size,
            resnet_layers,
            pool_kernel_size,
            pool_stride,
            with_pool=True
        ):
        output_size = input_size
        for _ in range(resnet_layers):
            output_size = self._calculate_output_size_conv(
                input_size=output_size,
                kernel_size=kernel_size,
                padding_size=padding_size,
                stride=stride
            )
        if with_pool:
            output_size = self._calculate_output_size_pool(
                input_size=output_size,
                kernel_size=pool_kernel_size,
                stride=pool_stride
            )
        return output_size
    
    def _calculate_output_size_conv(self, input_size, kernel_size, padding_size, stride):
        return ((input_size - kernel_size + 2 * padding_size) // stride) + 1
    
    def _calculate_output_size_pool(self, input_size, kernel_size, stride):
        return ((input_size - kernel_size) // stride) + 1

    def forward(self, x):
        x = self.input_conv(x)
        x = self.batch_norm_initial(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool_initial(x)
        identity = x
        for i in range(self.layers):
            if i == self.layers - 1:
                x = self.conv[i](x)
                x = self.batch_norm[i](x)
                if self.downsample:
                    identity = self.downsample_layer(identity)
                identity = [self.attention[j](identity[:,j,:,:], identity[:,j,:,:]) for j in range(len(self.attention))]
                identity = torch.stack(identity, dim=1).reshape(x.size(0), x.size(1), x.size(2), x.size(3))
                x = torch.matmul(identity, x)
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.conv[i](x)
                x = self.batch_norm[i](x)
                x = self.relu(x)
                x = self.dropout(x)
        x = self.pool_out(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(
            self, 
            input_size:int,
            input_channels:int,
            n_channels:int, 
            kernel_size:int, 
            stride:int, 
            padding:int,
            layers:int = 2,
            pool_kernel_size:int = 2,
            pool_stride:int = 2,
            dropout: float = 0.2,
        ):
        super(ResNetBlock, self).__init__()
        self.layers = layers
        self.input_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.batch_norm_initial = nn.BatchNorm2d(n_channels)
        self.pool_initial = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        output_size = self._calculate_output_size_conv(
            input_size=input_size,
            kernel_size=kernel_size,
            padding_size=padding,
            stride=stride
        )
        output_size = self._calculate_output_size_pool(
            input_size=output_size,
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        self.conv = nn.ModuleList([nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ) for i in range(layers)])
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm2d(n_channels) 
            for _ in range(layers)
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        input_size = output_size
        output_size = self._calculate_output_size_flow(
            input_size=output_size,
            kernel_size=kernel_size,
            padding_size=padding,
            stride=stride,
            resnet_layers=layers,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            with_pool=False,
        )
        if input_size != output_size:
            self.downsample = True
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=int(((input_size - 1) - (output_size - 1) * 1 + 2 * 0) / 1 + 1),
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(n_channels)
            )
        else:
            self.downsample = False
            self.downsample_layer = None
        self.pool_out = nn.MaxPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )
        self.output_size = self._calculate_output_size_pool(
            input_size=output_size,
            kernel_size=pool_kernel_size,
            stride=pool_stride
        )

    def _calculate_output_size_flow(
            self,
            input_size,
            kernel_size,
            stride,
            padding_size,
            resnet_layers,
            pool_kernel_size,
            pool_stride,
            with_pool=True
        ):
        output_size = input_size
        for _ in range(resnet_layers):
            output_size = self._calculate_output_size_conv(
                input_size=output_size,
                kernel_size=kernel_size,
                padding_size=padding_size,
                stride=stride
            )
        if with_pool:
            output_size = self._calculate_output_size_pool(
                input_size=output_size,
                kernel_size=pool_kernel_size,
                stride=pool_stride
            )
        return output_size
    
    def _calculate_output_size_conv(self, input_size, kernel_size, padding_size, stride):
        return ((input_size - kernel_size + 2 * padding_size) // stride) + 1
    
    def _calculate_output_size_pool(self, input_size, kernel_size, stride):
        return ((input_size - kernel_size) // stride) + 1

    def forward(self, x):
        x = self.input_conv(x)
        x = self.batch_norm_initial(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool_initial(x)
        identity = x
        for i in range(self.layers):
            if i == self.layers - 1:
                x = self.conv[i](x)
                x = self.batch_norm[i](x)
                if self.downsample:
                    identity = self.downsample_layer(identity)
                x += identity
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.conv[i](x)
                x = self.batch_norm[i](x)
                x = self.relu(x)
                x = self.dropout(x)
        x = self.pool_out(x)
        return x
    
class ResNet(nn.Module):

    def __init__(
            self,
            n_classes: int = 3,
            image_input_shape: tuple|list = (256, 256),
            input_channels: int = 3,
            resnet_blocks: int = 2,
            resnet_channels: list[int] = [6, 12],
            resnet_kernel_sizes: list[int] = [5, 5],
            resnet_strides: list[int] = [1, 1],
            resnet_padding_sizes: list[int] = [0, 0],
            resnet_layers: list[int] = [2, 2],
            pool_kernel_size: int = 2,
            pool_stride: int = 2,
            fc1_output_dims: int = 120,
            fc2_output_dims: int = 84,
            dropout: float = 0.2,
            resnet_block_class: nn.Module = ResNetBlock
        ):
        super(ResNet, self).__init__()
        self.resnet_block_class = resnet_block_class
        self.resnet_blocks, output_size = self._create_resnet_blocks(
            resnet_blocks=resnet_blocks,
            resnet_channels=resnet_channels,
            resnet_kernel_sizes=resnet_kernel_sizes,
            resnet_strides=resnet_strides,
            resnet_padding_sizes=resnet_padding_sizes,
            resnet_layers=resnet_layers,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            input_size=image_input_shape[0],
            input_channels=input_channels,
            dropout=dropout
        )
        fc1_input_dims = resnet_channels[-1] * output_size * output_size
        
        self.fc1 = nn.Linear(fc1_input_dims, fc1_output_dims)
        self.fc2 = nn.Linear(fc1_output_dims, fc2_output_dims)
        self.fc3 = nn.Linear(fc2_output_dims, n_classes)
        self.output_activation = nn.Softmax(dim=1)
        self.config = {
            "n_classes": n_classes,
            "image_input_shape": image_input_shape,
            "input_channels": input_channels,
            "resnet_blocks": resnet_blocks,
            "resnet_channels": resnet_channels,
            "resnet_kernel_sizes": resnet_kernel_sizes,
            "resnet_strides": resnet_strides,
            "resnet_padding_sizes": resnet_padding_sizes,
            "resnet_layers": resnet_layers,
            "pool_kernel_size": pool_kernel_size,
            "pool_stride": pool_stride,
            "fc1_output_dims": fc1_output_dims,
            "fc2_output_dims": fc2_output_dims,
            "dropout": dropout
        }

    def _create_resnet_blocks(
            self,
            resnet_blocks,
            resnet_channels,
            resnet_kernel_sizes,
            resnet_strides,
            resnet_padding_sizes,
            resnet_layers,
            pool_kernel_size,
            pool_stride,
            input_size,
            input_channels,
            dropout
        ):
        block_list = []
        for i in range(resnet_blocks):
            new_block = self.resnet_block_class(
                input_size=input_size,
                input_channels=input_channels,
                n_channels=resnet_channels[i],
                kernel_size=resnet_kernel_sizes[i],
                padding=resnet_padding_sizes[i],
                stride=resnet_strides[i],
                layers=resnet_layers[i],
                pool_kernel_size=pool_kernel_size,
                pool_stride=pool_stride,
                dropout=dropout
            )
            input_size = new_block.output_size
            input_channels = resnet_channels[i]
            block_list.append(new_block)
        return nn.ModuleList(block_list), input_size

    def _calculate_output_size_conv(self, input_size, kernel_size, padding_size, stride):
        return ((input_size - kernel_size + 2 * padding_size) // stride) + 1
    
    def _calculate_output_size_pool(self, input_size, kernel_size, stride):
        return ((input_size - kernel_size) // stride) + 1
    
    def forward(self, x):
        x = x / 255.0
        for block in self.resnet_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
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
    

class ViT(nn.Module):
    def __init__(
            self, 
            channels:int = 3, 
            x_dim:int = 255, 
            y_dim:int = 255, 
            n_patches:int = 51, 
            hidden_d:int = 128, 
            device:torch.device = torch.device("cpu")
        ):
        super(ViT, self).__init__()

        self.chw = (channels, x_dim, y_dim)
        self.n_patches = n_patches
        self.hidden_d = hidden_d

        assert self.chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert self.chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (self.chw[1] / n_patches, self.chw[2] / n_patches)

        self.input_d = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        self.register_buffer('positional_embeddings', self._get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        self.positional_embeddings.to(device)
        self.device = device

    def forward(self, images):
        n, c, h, w = images.shape
        patches = self._patchify(images).to(self.positional_embeddings.device)
        tokens = self.linear_mapper(patches)
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        return out
  
    def _patchify(self, images):
        n, c, h, w = images.shape

        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, self.n_patches ** 2, h * w * c // self.n_patches ** 2)
        patch_size = h // self.n_patches

        for idx, image in enumerate(images):
            for i in range(self.n_patches):
                for j in range(self.n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * self.n_patches + j] = patch.flatten()
        return patches
    
    def _get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
    
class ViTTransformer(nn.Module):
    def __init__(
            self,
            n_classes: int = 3,
            input_channels: int = 3,
            image_input_shape: tuple|list = (255, 255),
            hidden_d: int = 128,
            n_patches: int = 51,
            n_heads: int = 8,
            dropout: float = 0.2,
        ):
        super(ViTTransformer, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.vit = ViT(
            channels=input_channels, 
            x_dim=image_input_shape[0], 
            y_dim=image_input_shape[1], 
            n_patches=n_patches, 
            hidden_d=hidden_d,
            device=self.device
        )
        self.attention = nn.MultiheadAttention(hidden_d, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_d)
        self.output_linear = nn.Linear(hidden_d, n_classes)
        self.output_activation = nn.Softmax(dim=1)
        self.config = {
            "n_classes": n_classes,
            "input_channels": input_channels,
            "image_input_shape": image_input_shape,
            "hidden_d": hidden_d,
            "n_patches": n_patches,
            "n_heads": n_heads,
            "dropout": dropout
        }

    def forward(self, x):
        x = self.vit(x)
        x, _ = self.attention(x, x, x)
        x = self.norm(x)
        x = self.dropout(x)
        x = x.mean(dim=1)
        x = self.output_linear(x)
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