import os
from collections import OrderedDict
import argparse

import torch
import numpy as np
import yaml
from progress.bar import Bar

from utils import get_dataloader
from models.resnet_ours_cbam_multi import ResidualNet, classifier
from conf import settings, global_config as cfg

def get_model_list():
    num_env = 5
    split_layer = 3

    model_list = []
    for e in range(num_env):
        if (e <= num_env):
            model_list.append(classifier(num_classes=cfg.num_classes, bias=False))

    model_list.append(classifier(num_classes=cfg.num_classes))
    model_list.append(
        ResidualNet(
            cfg.network_type, 18, num_classes=cfg.num_classes, att_type='CBAM', split_layer=split_layer
        )
    )

    # Use GPU
    model_list = [ model_list_.cuda() for model_list_ in model_list ]

    return model_list


def update_key_names(d):
    new_d = OrderedDict()
    for old_key in d:
        new_key = '.'.join(old_key.split('.')[1:])
        new_d[new_key] = d[old_key]

    return new_d


def load_pretrained(path):
    nets = get_model_list()
    state_dict_dict = torch.load(path)

    for idx, net in enumerate(nets):
        #updated_state_dict = update_key_names(state_dict_dict[idx])
        #net.load_state_dict(updated_state_dict)
        net.load_state_dict(state_dict_dict[idx])

    print('Loaded pretrained model...')

    return nets


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compute representations of causal and correlational signals')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'], required=True)
    parser.add_argument('--trained_model', type=str, required=True)

    args = parser.parse_args()

    cfg.network_type = args.dataset.upper()
    if args.dataset == 'mnist':
        cfg.mean, cfg.std = settings.MNIST_TRAIN_MEAN, settings.MNIST_TRAIN_STD
        cfg.img_shape, cfg.load_size = (3, 28, 28), 28          # For simplicity just repeat first channel
        cfg.num_classes, cfg.num_samples = 10, 60_000
    elif args.dataset == 'cifar10':
        cfg.mean, cfg.std = settings.CIFAR10_TRAIN_MEAN, settings.CIFAR10_TRAIN_STD
        cfg.img_shape, cfg.load_size = (3, 32, 32), 32
        cfg.num_classes, cfg.num_samples = 10, 60_000
    elif args.dataset == 'cifar100':
        cfg.mean, cfg.std = settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD
        cfg.img_shape, cfg.load_size = (3, 32, 32), 32
        cfg.num_classes, cfg.num_samples = 100, 60_000
    else:
        raise ValueError(f'Dataset {args.dataset} is not supported')

    
    # Load pretrained model
    nets = load_pretrained(args.trained_model)
    model = nets[-1]

    # Use GPU and set to train mode to False
    model.cuda()
    model.eval()

    # Load dataset
    print('Loading test split')
    test_loader = get_dataloader(
        config={'dataset': args.dataset}, mean=cfg.mean, std=cfg.std, 
        num_workers=8, batch_size=128, train=False,
    )

    train_loader = get_dataloader(
        config={'dataset': args.dataset}, mean=cfg.mean, std=cfg.std, 
        num_workers=8, batch_size=128, train=True,
    )

    # Forward pass over all data to compute representations
    print('Starting representations computation')
    split_name2loader = { 'train': train_loader, 'test':test_loader }

    with torch.no_grad():
        for split_name, loader in split_name2loader.items():
            images, style, content = [], [], []
            
            print(f'Split: {split_name}')
            bar = Bar('Processing', max=len(loader), index=0)
            for (x, _) in loader:
                x = x.cuda()

                # Estimate causal and spurious signals
                x_causal, x_spurious, x_mix = model(x)

                # Store representations
                images.append(x)
                content.append(x_causal)
                style.append(x_spurious)

                bar.next()

            ## Save images, contents and styles for test and train sets
            images = torch.cat(images, 0).detach().cpu().numpy()
            style = torch.cat(style, 0).detach().cpu().numpy()
            content = torch.cat(content, 0).detach().cpu().numpy()

            outdir, checkpoint_name = os.path.split(args.trained_model)

            checkpoint_name = checkpoint_name.split('.')[0]
            outpath = os.path.join(outdir, f'representations-{checkpoint_name}')
            os.makedirs(outpath, exist_ok=True)
            np.savez(os.path.join(outpath, f'content_{split_name}.npz'), content)
            np.savez(os.path.join(outpath, f'images_{split_name}.npz'), images)
            np.savez(os.path.join(outpath, f'style_{split_name}.npz'), style)

    print('Representations computed and stored under {outpath}')