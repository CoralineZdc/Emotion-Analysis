import numpy as np
import random
from PIL import Image, ImageChops
import math
from typing import Optional, Tuple, Union, Sequence
import torch


class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for transform in self.transforms:
            if isinstance(img, (tuple, list)):
                img = tuple(transform(crop) for crop in img)
            else:
                img = transform(img)
        return img
    

class ToTensor:
    """Converts a PIL Image or numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]."""
    def __call__(self, pic: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(pic, np.ndarray):
            if pic.ndim == 2:
                pic = pic[:, :, np.newaxis]
            img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
            return img.float().div(255) if img.dtype == torch.uint8 else img.float()

        if not isinstance(pic, Image.Image):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))


        if pic.mode == "1": img = 255 * torch.from_numpy(np.array(pic, dtype=np.uint8))
        else: img = torch.from_numpy(np.array(pic, dtype=np.uint8))

        if pic.mode == "L": img = img.view(pic.size[1], pic.size[0], 1)
        else: img = img.view(pic.size[1], pic.size[0], len(pic.mode))

        # Convert HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        return img.float().div(255) if img.dtype == torch.uint8 else img.float()


class Normalize:
    """Normalize a tensor image with mean and standard deviation."""
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor, inplace: bool = False):
        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean_t = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std_t = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std_t == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))

        if mean_t.ndim == 1:
            mean_t = mean_t.view(-1, 1, 1)
        if std_t.ndim == 1:
            std_t = std_t.view(-1, 1, 1)

        tensor.sub_(mean_t).div_(std_t)
        return tensor
    

class Resize(object):
    """Resize the input PIL Image to the given size."""
    def __init__(self, size: Union[int, Sequence[int]], interpolation: int  =Image.BILINEAR) -> None:
        if isinstance(size, int):
            self.size: Union[int, Tuple[int, int]] = (size, size)
        elif isinstance(size, Sequence) and len(size) == 2:
            self.size = (int(size[0]), int(size[1]))
        else:
            raise ValueError("Size should be an int or a sequence of length 2.")

        self.interpolation = interpolation

    def __call__(self, img: Image.Image, interpolation: int = Image.BILINEAR):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
            return img.resize((ow, oh), interpolation)
        
        return img.resize(self.size[::-1], interpolation)

    

class RandomHorizontalFlip(object):
    """Horizontally flips the given PIL Image randomly with a probability of p (default 0.5)."""
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    
    
class RandomRotation(object):
    """Rotate the image by a random anfle within [-degrees, +degrees]. """

    def __init__(
            self, 
            degrees: Union[float, Sequence[float]], 
            resample: int = Image.NEAREST, 
            expand: bool = False, center: 
            Optional[Tuple[float, float]] = None
        ) -> None:
        if isinstance(degrees, (int, float)):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        elif isinstance(degrees, Sequence) and len(degrees) == 2:
            self.degrees = (float(degrees[0]), float(degrees[1]))
        else:
            raise ValueError("Degrees should be a single number or a sequence of length 2.")

        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, img):
        angle = np.random.uniform(self.degrees[0], self.degrees[1])
        return img.rotate(angle, resample=self.resample, expand=self.expand, center=self.center)


class RandomResizedCrop:
    """Crops the image to a random area and aspect ratio, then resizes it to target size."""
    def __init__(
        self, 
        size: Union[int, Sequence[int]], 
        scale: Tuple[float, float] = (0.95, 1.05), 
        ratio: Tuple[float, float] = (1.0, 1.0), 
        interpolation: int = Image.BILINEAR
    ) -> None:
        if isinstance(size, int):
            self.size: Union[int, Tuple[int, int]] = size
        else:
            values = tuple(size)
            if len(values) != 2:
                raise ValueError("size must contain exactly two integers")
            self.size = (int(values[0]), int(values[1]))

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def _get_params(self, img: Image.Image) -> Tuple[int, int, int, int]:
        area = img.size[0] * img.size[1]
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            width = int(round(math.sqrt(target_area * aspect_ratio)))
            height = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                width, height = height, width

            if width <= img.size[0] and height <= img.size[1]:
                down_side = random.randint(0, img.size[1] - height)
                right_side = random.randint(0, img.size[0] - width)
                return down_side, right_side, height, width

        # Fallback crop
        width = min(img.size[0], img.size[1])
        down_side = (img.size[1] - width) // 2
        right_side = (img.size[0] - width) // 2
        return down_side, right_side, width, width

    def __call__(self, img: Image.Image) -> Image.Image:
            down_side, right_side, height, width = self._get_params(img)
            img = img.crop((right_side, down_side, right_side + width, down_side + height))
            if isinstance(self.size, int):
                if (width <= height and width == self.size) or (height <= width and height == self.size):
                    return img
                if width < height:
                    new_width = self.size
                    new_height = int(self.size * height / width)
                else:
                    new_height = self.size
                    new_width = int(self.size * width / height)
                return img.resize((new_width, new_height), self.interpolation)
            new_width, new_height = self.size
            return img.resize((new_width, new_height), self.interpolation)
    

class RandomShift(object):
    """Shift the given PIL Image randomly."""

    def __init__(self, shift_range = (0.05, 0.05)):
        self.shift_range = shift_range

    def __call__(self, img):
        max_shift_x = int(self.shift_range[0] * img.size[0])
        max_shift_y = int(self.shift_range[1] * img.size[1])

        shift_x = random.randint(-max_shift_x, max_shift_x)
        shift_y = random.randint(-max_shift_y, max_shift_y)
        return ImageChops.offset(img, shift_x, shift_y)
