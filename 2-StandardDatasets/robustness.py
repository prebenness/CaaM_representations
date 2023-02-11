import os
import argparse

import torch
import torchattacks


from representations import load_pretrained
from conf import settings, global_config as cfg
from dataset_utils import get_generic_dataloader

class CaaMWrapper(torch.nn.Module):
    def __init__(self, nets):
        super().__init__()

        # Set eval mode and GPU
        [ net.cuda() for net in nets ]
        [ net.eval() for net in nets ]

        # Linear classifiers
        self.linears = nets[:-1]
        self.linear_w = torch.stack([net.fc.weight for net in self.linears], 0).mean(0)

        # Image embedder
        self.embedder = nets[-1]
    
    def forward(self, x):
        x_causal, _, _ = self.embedder(x)
        y_pred = torch.nn.functional.linear(x_causal, self.linear_w)

        return y_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Compute representations of causal and correlational signals')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'], required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--trained_model', type=str, required=True)
    parser.add_argument('--no_gpu', action='store_true', required=False)


    args = parser.parse_args()

    cfg.device = 'cpu' if args.no_gpu else 'cuda'
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

    
    # Load dataset
    test_loader = get_generic_dataloader(
        os.path.join(args.data_root, 'test'), batch_size=128,
        train=False, val_data='NOT_IMAGENET'
    )
    
    num_corr, num_corr_adv, num_tot = 0, 0, 0
    for (x, y, _) in test_loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        # Load pretrained model
        nets = load_pretrained(args.trained_model)    
        model = CaaMWrapper(nets)

        # Get clean test preds
        y_pred = model(x)
        
        # Craft adversarial samples
        attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        attack.set_normalization_used(mean=cfg.mean, std=cfg.std)
        x_adv = attack(x, y)
        y_pred_adv = model(x_adv)

        num_corr += (y_pred.argmax(dim=1) == y).sum().item()
        num_corr_adv += (y_pred_adv.argmax(dim=1) == y).sum().item()
        num_tot += y.shape[0]

        print(f'Processed {num_tot:5d} of {len(test_loader.dataset):5d} samples, current tally: Clean acc: {num_corr / num_tot:4.3f} Adv acc: {num_corr_adv / num_tot:4.3f}')

    # Compute and store results
    clean_acc = num_corr / num_tot
    adv_acc = num_corr_adv / num_tot

    res_text = f'Test completed: Clean acc: {clean_acc} Adv acc: {adv_acc}'

    model_dir, model_file = os.path.split(args.trained_model)
    log_file_name = '.'.join([ model_file.split('.')[0], 'txt' ])
    with open(os.path.join(model_dir, log_file_name), 'w') as w:
        w.write(res_text) 

    print(res_text)