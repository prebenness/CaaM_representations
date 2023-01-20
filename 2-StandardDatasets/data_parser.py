#### Shitty hack to get around the abhorrent dataset format used

import os

import torch
import torchvision


def main():
    # Config
    name, num_classes, make_debug_set = 'mnist', 10, False

    output_dir = os.path.join('data', f'{name}_dumb{"_debug" if make_debug_set else ""}')
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_dataloaders(name)
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
        for class_idx in range(num_classes):
            class_path = os.path.join(split_output_dir, f'{class_idx}_class')
            class_paths[class_idx] = class_path
            os.makedirs(class_path, exist_ok=True)

        count = 0           # Need some way of giving unique filenames
        batch_count = 1     # Progress tracking for sanity

        if make_debug_set:
            classes_stored = { idx: False for idx in range(num_classes) }

        for x_batch, y_batch in loader:
            for x, y in zip(x_batch, y_batch):
                count += 1
                
                class_idx = y.item()        # Access single item of 0-dim tensor
                file_path = os.path.join(class_paths[class_idx], f'image_{count}.JPEG')

                # For debug sets, only store one sample per class
                if make_debug_set:
                    if not classes_stored[class_idx]:
                        torchvision.utils.save_image(x, file_path, format='JPEG')
                        classes_stored[class_idx] = True
                    
                    if all(classes_stored.values()):
                        print('Debug split created, breaking')
                        debug_split_done = True
                        break
                else:
                    torchvision.utils.save_image(x, file_path, format='JPEG')
            
            if make_debug_set and debug_split_done:
                break

            print(f'Parsed batch {batch_count} of {len(loader)}')
            batch_count += 1
    
    print(f'Finished parsing images - results stored under {output_dir}')


def make_dataloaders(name):
    if name == 'mnist':
        Dataset = torchvision.datasets.MNIST
        split = (50_000, 10_000)
    elif name == 'cifar10':
        Dataset = torchvision.datasets.CIFAR10
        split = (40_000, 10_000)
    elif name == 'cifar100':
        Dataset = torchvision.datasets.CIFAR100
        split = (40_000, 10_000)
    else:
        raise ValueError(f'Dataset {name} not supported')

    # Load datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Resize(224)              # ImageNet format is 224x224 px
    ])
    train_dataset = Dataset(root='data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, split)
    test_dataset = Dataset(root='data', train=False, download=True, transform=transform)

    def loader_factory(d):
        return torch.utils.data.DataLoader(
            dataset=d, batch_size=128, shuffle=False, num_workers=4,
            pin_memory=True
        )

    # Make dataloaders
    train_dataloader = loader_factory(train_dataset)
    val_dataloader = loader_factory(val_dataset)
    test_dataloader = loader_factory(test_dataset)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    main()