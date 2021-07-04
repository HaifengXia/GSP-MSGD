import torchvision.transforms as transforms
from PIL import Image
import torch

def get_transform(train=True):
    transform_list = []
    osize = [256, 256]
    transform_list.append(transforms.Resize(osize, Image.BICUBIC))
    if train:
        transform_list.append(transforms.RandomCrop(224))
    else:
        transform_list.append(transforms.CenterCrop(224))

    if train:
        transform_list.append(transforms.RandomHorizontalFlip())

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                       std=(0.229, 0.224, 0.225))]


    transform_list += to_normalized_tensor

    return transforms.Compose(transform_list)
