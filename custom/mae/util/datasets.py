import os
import PIL
import numpy as np

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .custom_datasets import CustomImageDataset, CustomHeterogeneousDataset

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.heterogeneous:
        dataset = CustomHeterogeneousDataset(is_train, args=args, transform=transform)
    else:
        dataset = CustomImageDataset(is_train, args=args, transform=transform)
    # dataset = datasets.CIFAR10(
    #     root='./cifar10',
    #     train=is_train,
    #     transform=transform, 
    #     #把灰階從0~255壓縮到0~1
    #     download=True
    # )

    print(dataset)

    return dataset

class BroadcastTo3CH:
    def __call__(self, sample):
        sample = np.array(sample)
        shape = sample.shape
        if len(shape) == 2:
            h, w = shape
            sample = np.expand_dims(sample, -1)
            sample = np.broadcast_to(sample, (h, w, 3))
            sample = PIL.Image.fromarray(sample)
        return sample
    
def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # mean = 0
    # std = 1
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train

        # transform = create_transform(
        #     input_size=args.input_size,
        #     is_training=True,
        #     color_jitter=args.color_jitter,
        #     auto_augment=args.aa,
        #     interpolation='bicubic',
        #     re_prob=args.reprob,
        #     re_mode=args.remode,
        #     re_count=args.recount,
        #     mean=mean,
        #     std=std,
        # )
        transform = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.crop_size),
            transforms.RandomEqualize(p=1),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1)),
            # transforms.Normalize(mean, std),
            transforms.ToTensor(),
        ])

        return transform

    # eval transform
    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.crop_size),
        transforms.RandomEqualize(p=1),
        # transforms.Normalize(mean, std),
        transforms.ToTensor(),
    ])
    return test_transform