import torch
from skimage import transform
import numpy as np


class ImageTransform:

    def __init__(
            self,
            output_size: tuple[int, int],
            rescale: bool = True,
            random_crop: bool|float = 0.2,
            to_tensor: bool = True,
        ):
        self.rescale = rescale
        self.random_crop = random_crop
        self.to_tensor = to_tensor
        self.output_size = output_size

    def _load_data(self, sample: dict)->dict:
        if isinstance(sample["image"], torch.Tensor):
            image = sample["image"].numpy().transpose((1, 2, 0))
            label = sample["label"]
        else:
            image, label = sample['image'], sample['label']
        return {
            'image': image,
            'label': label
        }

    def _rescale(self, sample: dict)-> dict:
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'label': label}
    
    def _random_crop(self, sample: dict)-> dict:
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}
    
    def _to_tensor(self, sample: dict)-> dict:
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': label}
    
    def __call__(self, sample: dict)-> dict:
        sample = self._load_data(sample)
        if self.rescale:
            sample = self._rescale(sample)
        if self.random_crop:
            if isinstance(self.random_crop, float):
                if np.random.rand() < self.random_crop:
                    sample = self._random_crop(sample)
            else:
                sample = self._random_crop(sample)
        if self.to_tensor:
            sample = self._to_tensor(sample)
        return sample
    
    def no_augment(self):
        return ImageTransform(
            output_size=self.output_size, 
            rescale=self.rescale, 
            random_crop=False, 
            to_tensor=True
        )