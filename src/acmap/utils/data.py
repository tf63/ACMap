import os

import numpy as np
from torchvision import datasets, transforms

from acmap.utils.toolkit import split_images_labels


class iData:
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(f'{self.dataset_dir}', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(f'{self.dataset_dir}', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(f'{self.dataset_dir}', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(f'{self.dataset_dir}', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


def build_transform_coda_prompt(is_train, dataset):
    if is_train:
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]
        return transform

    t = []
    if dataset.startswith('imagenet'):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ]

    return t


def build_transform(is_train):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    # return transforms.Compose(t)
    return t


class iCIFAR224(iData):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir=dataset_dir)
        self.use_path = False

        self.train_trsf = build_transform(True)
        self.test_trsf = build_transform(False)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(f'{self.dataset_dir}', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(f'{self.dataset_dir}', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, 'You should specify the folder of your dataset'
        train_dir = os.path.join(self.dataset_dir, 'imagenet', 'train')
        test_dir = os.path.join(self.dataset_dir, 'imagenet', 'val')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, 'You should specify the folder of your dataset'
        train_dir = os.path.join(self.dataset_dir, 'imagenet', 'train')
        test_dir = os.path.join(self.dataset_dir, 'imagenet', 'val')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir=dataset_dir)
        self.use_path = True

        self.train_trsf = build_transform(True)
        self.test_trsf = build_transform(False)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = os.path.join(self.dataset_dir, 'imagenet-r', 'train')
        test_dir = os.path.join(self.dataset_dir, 'imagenet-r', 'test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True

    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = os.path.join(self.dataset_dir, 'imagenet-a', 'train')
        test_dir = os.path.join(self.dataset_dir, 'imagenet-a', 'test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class CUB(iData):
    use_path = True

    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = os.path.join(self.dataset_dir, 'cub', 'train')
        test_dir = os.path.join(self.dataset_dir, 'cub', 'test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True

    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    common_trsf = []

    class_order = np.arange(200).tolist()

    def download_data(self):
        assert 0, 'You should specify the folder of your dataset'
        train_dir = os.path.join(self.dataset_dir, 'objectnet', 'train')
        test_dir = os.path.join(self.dataset_dir, 'objectnet', 'test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True

    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    common_trsf = []

    class_order = np.arange(300).tolist()

    def download_data(self):
        assert 0, 'You should specify the folder of your dataset'
        train_dir = os.path.join(self.dataset_dir, 'omnibenchmark', 'train')
        test_dir = os.path.join(self.dataset_dir, 'omnibenchmark', 'test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class vtab(iData):
    use_path = True

    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    common_trsf = []

    class_order = np.arange(50).tolist()

    def download_data(self):
        train_dir = os.path.join(self.dataset_dir, 'vtab', 'train')
        test_dir = os.path.join(self.dataset_dir, 'vtab', 'test')

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
