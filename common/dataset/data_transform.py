from torchvision import transforms

train_tf_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_tf_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

def get_transform(dataset_name: str, split: str = "train"):
    """
    根据 dataset_name 和 split 返回 transform
    :param dataset_name: 数据集名字，例如 "cifar10", "nwpu", "dota"
    :param split: "train" 或 "test"
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return train_tf_cifar10 if split == "train" else test_tf_cifar10
    # elif dataset_name == "nwpu":
    #     return train_tf_nwpu if split == "train" else test_tf_nwpu
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")