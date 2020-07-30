import torch
import torchvision
import torchvision.transforms as transforms


def get_loader(data_path, batch_size, mode, num_workers=4):

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if mode == 'train':
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    else:
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                               download=True, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

    return dataloader
