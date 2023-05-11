#### Shitty hack to get around the abhorrent dataset format used

import os
import argparse

import torch
import torchvision
from torch.utils.data import random_split


def main(dataset_config, debug=False):

    output_dir = os.path.join('data', f'{dataset_config["name"]}_dumb{"_debug" if debug else ""}')
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data(dataset_config)
    split_name2loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    
    for split_name, loader in split_name2loader.items():
        debug_split_done = False

        print(f'Parsing images in {split_name} split')

        # Make output directory for split
        split_output_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # Make sub-directories for each class: split/idx_class
        class_paths = {}
        for class_idx in range(dataset_config['num_classes']):
            class_path = os.path.join(split_output_dir, f'{class_idx}_class')
            class_paths[class_idx] = class_path
            os.makedirs(class_path, exist_ok=True)

        count = 0           # Need some way of giving unique filenames
        batch_count = 1     # Progress tracking for sanity

        if debug:
            classes_stored = { idx: False for idx in range(dataset_config['num_classes']) }

        for x_batch, y_batch in loader:
            for x, y in zip(x_batch, y_batch):
                count += 1
                
                class_idx = y.item()        # Access single item of 0-dim tensor
                file_path = os.path.join(class_paths[class_idx], f'image_{count}.JPEG')

                # For debug sets, only store one sample per class
                if debug:
                    if not classes_stored[class_idx]:
                        torchvision.utils.save_image(x, file_path, format='JPEG')
                        classes_stored[class_idx] = True
                    
                    if all(classes_stored.values()):
                        print('Debug split created, breaking')
                        debug_split_done = True
                        break
                else:
                    torchvision.utils.save_image(x, file_path, format='JPEG')
            
            if debug and debug_split_done:
                break

            print(f'Parsed batch {batch_count} of {len(loader)}')
            batch_count += 1
    
    print(f'Finished parsing images - results stored under {output_dir}')


def country211_getter(*args, train=True, **kwargs):
    '''
    Wrapper function for the Country-211 dataset, emulates 
    behaviour of other dataset getters
    '''
    if train:
        train_split = torchvision.datasets.Country211(
            *args, split='train', **kwargs
        )
        val_split = torchvision.datasets.Country211(
            *args, split='valid', **kwargs
        )
        dataset = torch.utils.data.ConcatDataset([train_split, val_split])
    else:
        dataset = torchvision.datasets.Country211(
            *args, split='test', **kwargs
        )

    return dataset


def pcam_getter(*args, train=True, ** kwargs):
    '''
    Wrapper function for the PCAM dataset, emulates 
    behaviour of other dataset getters
    '''
    if train:
        train_split = torchvision.datasets.PCAM(
            *args, split='train', **kwargs
        )
        val_split = torchvision.datasets.PCAM(
            *args, split='val', **kwargs
        )
        dataset = torch.utils.data.ConcatDataset([train_split, val_split])
    else:
        dataset = torchvision.datasets.PCAM(
            *args, split='test', **kwargs
        )

    return dataset


def get_data(dataset_config):
    '''
    Get a given supported dataset and return both clean and perturbed data
    samples
    '''

    if dataset_config['name'] == 'mnist':
        getter = torchvision.datasets.MNIST
    elif dataset_config['name'] == 'emnist_balanced':
        getter = lambda *args, **kwargs: torchvision.datasets.EMNIST(
            *args, split='balanced', **kwargs
        )
    elif dataset_config['name'] == 'fashion_mnist':
        getter = torchvision.datasets.FashionMNIST
    elif dataset_config['name'] == 'cifar10':
        getter = torchvision.datasets.CIFAR10
    elif dataset_config['name'] == 'cifar100':
        getter = torchvision.datasets.CIFAR100
    elif dataset_config['name'] == 'pcam':
        getter = pcam_getter
    elif dataset_config['name'] == 'imagenet':
        getter = torchvision.datasets.ImageNet
    elif dataset_config['name'] == 'country211':
        getter = country211_getter
    else:
        raise NotImplementedError(
            f'Dataset {dataset_config["name"]} not supported'
        )

    # Load datasets
    def transform(x):
        '''
        Transforms and manipulations to apply to images
        '''
        x = torchvision.transforms.ToTensor()(x)  # Pixels to range [0, 1]
        x = torchvision.transforms.Resize(size=(dataset_config['out_shape'][1:]))(x)
        x = torchvision.transforms.CenterCrop(size=(dataset_config['out_shape'][1:]))(x)

        return x

    def dataset_factory(train=True):
        return getter(
            root=os.path.join('data'),
            transform=transform, download=True,
            train=train
        )

    train_dataset = dataset_factory(train=True)
    test_dataset = dataset_factory(train=False)

    # Train val split:
    train_dataset, val_dataset = random_split(train_dataset, [4/5, 1/5])

    # Create PyTorch DataLoaders
    def dataloader_factory(dataset):
        return torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True,
            num_workers=8,
        )

    train_loader = dataloader_factory(train_dataset)
    val_loader = dataloader_factory(val_dataset)
    test_loader = dataloader_factory(test_dataset)

    # Return loaders for train, val, test, for clean, pert data
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data parsing script')
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    
    arrgs = parser.parse_args()

    dataset_configs = {
        'mnist': {
            'out_shape': (1, 28, 28),
            'num_classes': 10,
        },
        'emnist_balanced': {
            'out_shape': (1, 28, 28),
            'num_classes': 47,
        },
        'fashion_mnist': {
            'out_shape': (1, 28, 28),
            'num_classes': 10,
        },
        'cifar10': {
            'out_shape': (3, 32, 32),
            'num_classes': 10,
        },
        'cifar100': {
            'out_shape': (3, 32, 32),
            'num_classes': 100,
        },
        'country211': {
            'out_shape': (3, 224, 224),
            'num_classes': 211,
        },
        'pcam': {
            'out_shape': (3, 96, 96),
            'num_classes': 2,
        }
    }

    config = {
        'name': arrgs.dataset, **dataset_configs[arrgs.dataset]
    }

    main(config, debug=arrgs.debug)


